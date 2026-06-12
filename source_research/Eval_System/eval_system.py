"""Đánh giá end-to-end hệ thống Agentic RAG HIỆN TẠI.

Khác hệ thống cũ:
  - Cũ: gọi RAG.get_gemini_response_rag_final(...) (pipeline tuyến tính, Gemini,
    rerank Cohere) + embedding Gemini text-embedding-004 để tính cosine.
  - Nay: chạy Orchestrator (Planner → Retrieve → Grade → Generate → Check) qua
    SSE, lấy event `final` làm câu trả lời; tính cosine bằng CHÍNH embedding
    nội bộ của hệ thống (bhuy71/embedding_finetuned) cho nhất quán; ROUGE-L + F1
    dùng rouge-score chính thống.

Pipeline:
  1) `run`  : với mỗi câu hỏi (legal) → orchestrator.run(...) lấy answer → cache CSV.
  2) `score`: đọc cache → cosine + ROUGE-L + token-F1 vs đáp án vàng → báo cáo
              tổng + theo domain.
  3) `all`  : chạy cả hai.

Cách dùng (cần AWS credentials cho Bedrock + Qdrant đang chạy):
    python eval_system.py all
    python eval_system.py run --limit 20          # thử nhanh 20 mẫu
    python eval_system.py run --web               # bật web fallback khi RAG thiếu
    python eval_system.py score                   # chỉ tính metric từ cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = HERE / "eval_system_dataset.csv"
CACHE = HERE / "eval_system_predictions.csv"
REPORT = HERE / "eval_system_report.json"


# ============================================================
# 1) RUN — chạy orchestrator, cache câu trả lời
# ============================================================

def _build_orchestrator():
    """Khởi tạo Orchestrator giống hệt cách chat.py wiring (đủ stack thật)."""
    from source.core.config import Settings
    from source.model.generate_model import Gemini
    from source.model.embedding_model import Sentences_Transformer_Embedding
    from source.model.extract_model import Bert_Extract
    from source.model.rerank_model_finetune import RerankModelFinetune
    from source.data.vectordb.qdrant import Qdrant_Vector
    from source.search.utils_search import Qdrant_Utils
    from source.rerank.utils_rerank import Rerank_Utils
    from source.extract.utils_extract import Extract_Information
    from source.generate.generate import Gemini_Generate
    from source.tool.tavily_search import TavilySearchTool
    from source.tool.utils_search import Utils_Search_Tools
    from source.data.db.db_connection import DBConnection
    from source.data.db.memory import Memory
    from source.agent.planner import PlannerAgent
    from source.agent.grader import GraderAgent
    from source.agent.hallucination_checker import HallucinationChecker
    from source.agent.orchestrator import Orchestrator

    setting = Settings()
    gemini = Gemini(setting)
    bert = Bert_Extract(setting)
    embedding = Sentences_Transformer_Embedding(setting)
    qdrant = Qdrant_Vector(setting, embedding)
    model_finetune = RerankModelFinetune(setting)
    rerank_utils = Rerank_Utils(model_finetune)
    extract_utils = Extract_Information(bert)
    generate_utils = Gemini_Generate(gemini, setting)
    qdrant_utils = Qdrant_Utils(qdrant, generate_utils)
    web_tool = TavilySearchTool(setting)
    search_tools = Utils_Search_Tools(setting, generate_utils, extract_utils, web_tool)
    db = DBConnection(setting)
    memory = Memory(db, generator=generate_utils)

    planner = PlannerAgent(generate_utils)
    grader = GraderAgent(generate_utils, threshold=setting.GRADER_THRESHOLD)
    checker = HallucinationChecker(generate_utils)
    orch = Orchestrator(
        planner=planner, grader=grader, checker=checker, generator=generate_utils,
        qdrant_utils=qdrant_utils, rerank_utils=rerank_utils, extract_utils=extract_utils,
        search_tools=search_tools, memory=memory, settings=setting, embedding=embedding,
        max_iterations=setting.MAX_ITERATIONS,
    )
    return orch


def _run_one(orch, question: str, web: bool):
    """Chạy 1 câu hỏi qua orchestrator (không session) → (answer, intent, tool, n_refs, ms)."""
    answer, intent, tool, n_refs = "", "", "", 0
    t0 = time.time()
    for frame in orch.run(question, session_id=None, web_search_enabled=web):
        payload = frame[len("data: "):].strip()
        if not payload:
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        et = ev.get("type")
        if et == "plan":
            intent, tool = ev.get("intent", ""), ev.get("tool", "")
        elif et == "final":
            answer = ev.get("answer", "") or ""
            n_refs = len(ev.get("references", []) or [])
    return answer, intent, tool, n_refs, int((time.time() - t0) * 1000)


def run_predictions(limit: int = 0, web: bool = False, sample: int = 0, resume: bool = True):
    from tqdm import tqdm
    df = pd.read_csv(DATASET)

    # Lấy mẫu ngẫu nhiên cân bằng theo domain (nếu yêu cầu).
    # Nếu một domain không đủ quota, phần thiếu được bù sang các domain khác.
    if sample and sample > 0:
        import math
        rng_seed = 42
        domains = sorted(df["domain"].unique())
        groups = {d: df[df["domain"] == d] for d in domains}
        base = sample // len(domains)
        quota = {d: min(base, len(groups[d])) for d in domains}
        # bù phần còn thiếu (do domain nhỏ) sang domain còn dư, vòng tròn
        remaining = sample - sum(quota.values())
        while remaining > 0:
            progressed = False
            for d in domains:
                if remaining <= 0:
                    break
                if quota[d] < len(groups[d]):
                    quota[d] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break  # đã lấy hết tất cả mẫu
        parts = [groups[d].sample(quota[d], random_state=rng_seed) for d in domains]
        df = pd.concat(parts).sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    elif limit and limit > 0:
        df = df.head(limit).copy()

    # Resume: nếu đã có cache, bỏ qua các câu đã trả lời (system_answer khác rỗng)
    done = {}
    if resume and CACHE.exists():
        prev = pd.read_csv(CACHE)
        for _, r in prev.iterrows():
            ans = str(r.get("system_answer", "") or "")
            if ans.strip():
                done[str(r["question"])] = r.to_dict()
        print(f"[resume] đã có {len(done)} câu trả lời trước đó, sẽ bỏ qua.")

    orch = _build_orchestrator()
    rows = []
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc="Chạy hệ thống"):
        q = row["question"]
        if q in done:
            rows.append(done[q])
            continue
        try:
            a, it, tl, nr, ms = _run_one(orch, q, web)
        except Exception as e:
            print(f"\n[!] lỗi với câu: {q[:80]} :: {type(e).__name__}: {e}", flush=True)
            a, it, tl, nr, ms = "", "ERROR", "ERROR", 0, 0
        rec = {
            "domain": row["domain"], "question": q, "answer": row["answer"],
            "system_answer": a, "plan_intent": it, "plan_tool": tl,
            "n_references": nr, "latency_ms": ms,
        }
        rows.append(rec)
        # lưu tăng dần sau mỗi câu → không mất kết quả nếu dừng giữa chừng
        pd.DataFrame(rows).to_csv(CACHE, index=False)

    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"\n✓ Đã lưu {len(rows)} câu trả lời vào {CACHE.name}")
    return pd.DataFrame(rows)


# ============================================================
# 2) SCORE — cosine + ROUGE-L + F1
# ============================================================

def _make_embedder():
    from source.core.config import Settings
    from source.model.embedding_model import Sentences_Transformer_Embedding
    setting = Settings()
    return Sentences_Transformer_Embedding(setting)


def score(df: pd.DataFrame) -> dict:
    from rouge_score import rouge_scorer
    from sklearn.metrics.pairwise import cosine_similarity

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rouge_l(pred, ref):
        try:
            return scorer.score(str(ref), str(pred))["rougeL"].fmeasure
        except Exception:
            return 0.0

    def token_f1(pred, ref):
        pt = str(pred).lower().split()
        rt = str(ref).lower().split()
        if not pt or not rt:
            return 0.0
        common = set(pt) & set(rt)
        if not common:
            return 0.0
        p = len(common) / len(pt)
        r = len(common) / len(rt)
        return 2 * p * r / (p + r)

    # lọc câu trả lời rỗng / chỉ chứa khoảng trắng để tính metric
    df = df.copy()
    df["system_answer"] = df["system_answer"].fillna("").astype(str)
    valid = df[df["system_answer"].str.strip().str.len() > 0].copy()

    emb = _make_embedder()
    ref_vecs = emb.embeddings_bkai.embed_documents(valid["answer"].astype(str).tolist())
    sys_vecs = emb.embeddings_bkai.embed_documents(valid["system_answer"].tolist())
    cos = [float(cosine_similarity([r], [s])[0][0]) for r, s in zip(ref_vecs, sys_vecs)]
    valid["cosine"] = cos
    valid["rouge_l"] = valid.apply(lambda r: rouge_l(r["system_answer"], r["answer"]), axis=1)
    valid["token_f1"] = valid.apply(lambda r: token_f1(r["system_answer"], r["answer"]), axis=1)

    # ghi lại cache có metric
    merged = df.merge(valid[["question", "cosine", "rouge_l", "token_f1"]],
                      on="question", how="left")
    merged.to_csv(CACHE, index=False)

    def agg(sub):
        return {
            "n": int(len(sub)),
            "cosine": float(sub["cosine"].mean()) if len(sub) else 0.0,
            "rouge_l": float(sub["rouge_l"].mean()) if len(sub) else 0.0,
            "token_f1": float(sub["token_f1"].mean()) if len(sub) else 0.0,
        }

    report = {
        "n_total": int(len(df)),
        "n_scored": int(len(valid)),
        "n_empty_answer": int((df["system_answer"].str.strip().str.len() == 0).sum()),
        "overall": agg(valid),
        "by_domain": {d: agg(valid[valid["domain"] == d]) for d in sorted(valid["domain"].unique())},
    }
    # thống kê routing nếu có
    if "plan_tool" in df.columns:
        report["routing"] = df["plan_tool"].value_counts().to_dict()
        report["intent"] = df["plan_intent"].value_counts().to_dict()
    return report


def print_report(rep: dict):
    line = "=" * 60
    print("\n" + line)
    print("ĐÁNH GIÁ END-TO-END HỆ THỐNG AGENTIC RAG (HIỆN TẠI)")
    print(line)
    print(f"Tổng số câu hỏi   : {rep['n_total']}")
    print(f"Có câu trả lời    : {rep['n_scored']}")
    print(f"Câu trả lời rỗng  : {rep['n_empty_answer']}")
    o = rep["overall"]
    print("\n[TỔNG THỂ]")
    print(f"  Cosine similarity : {o['cosine']:.4f}")
    print(f"  ROUGE-L           : {o['rouge_l']:.4f}")
    print(f"  Token-F1          : {o['token_f1']:.4f}")
    print("\n[THEO DOMAIN]")
    print(f"  {'domain':<20}{'n':>5}{'cosine':>10}{'rougeL':>10}{'F1':>10}")
    for d, m in rep["by_domain"].items():
        print(f"  {d:<20}{m['n']:>5}{m['cosine']:>10.4f}{m['rouge_l']:>10.4f}{m['token_f1']:>10.4f}")
    if "routing" in rep:
        print("\n[ĐỊNH TUYẾN PLANNER]")
        print("  tool  :", rep["routing"])
        print("  intent:", rep["intent"])
    print(line)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Eval end-to-end hệ thống Agentic RAG hiện tại")
    ap.add_argument("mode", choices=["run", "score", "all"], nargs="?", default="all")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số câu (lấy N câu đầu)")
    ap.add_argument("--sample", type=int, default=0, help="Lấy mẫu ngẫu nhiên cân bằng theo domain (~N câu)")
    ap.add_argument("--web", action="store_true", help="Bật web fallback khi RAG thiếu")
    ap.add_argument("--no-resume", action="store_true", help="Không resume từ cache, chạy lại từ đầu")
    args = ap.parse_args()

    if args.mode in ("run", "all"):
        run_predictions(limit=args.limit, web=args.web, sample=args.sample,
                        resume=not args.no_resume)

    if args.mode in ("score", "all"):
        if not CACHE.exists():
            print(f"Chưa có cache {CACHE.name}. Chạy `python eval_system.py run` trước.")
            sys.exit(1)
        df = pd.read_csv(CACHE)
        rep = score(df)
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print_report(rep)
        print(f"✓ Đã lưu report vào {REPORT.name}")


if __name__ == "__main__":
    main()
