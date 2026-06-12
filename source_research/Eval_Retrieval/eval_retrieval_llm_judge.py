"""Đánh giá RETRIEVAL bằng LLM-as-judge (giải quyết vấn đề incomplete gold).

Khác eval_retrieval.py (khớp cứng 1 gold (SoHieu, Article)):
  - Ở đây mỗi tài liệu trong top-5 được Claude chấm "có chứa thông tin trả lời
    câu hỏi hay không" (yes/no). Một tài liệu được coi là RELEVANT theo phán
    quyết của judge, KHÔNG cần trùng đúng Điều luật gold.
  - Nhờ đó câu hỏi chung chung (nhiều điều luật cùng đúng) không bị tính
    false-negative như cách khớp cứng.

Pipeline (đúng như orchestrator._retrieve_and_rerank):
    generate_query (3 biến thể) → ViTokenizer → Qdrant(top_k=25, filter)
    → RRF → top 50 → Cross-Encoder rerank → top 5
Sau đó: với mỗi (query, doc top-5) → 1 lệnh gọi Claude judge.

Metric (binary relevance theo judge, trên top-5):
    Hit@1/3/5, MRR@5, NDCG@5, Precision@5  (+ trung bình theo domain / loại câu)

Cách dùng (cần Qdrant + AWS Bedrock + HF login):
    python eval_retrieval_llm_judge.py all --limit 100
    python eval_retrieval_llm_judge.py run --limit 100      # chỉ retrieval + judge, cache
    python eval_retrieval_llm_judge.py score                # tính metric từ cache
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = HERE / "retrieval_eval_dataset.json"
CACHE = HERE / "retrieval_judge_predictions.json"
REPORT = HERE / "retrieval_judge_report.json"

TOP_K = 5  # chỉ chấm top-5 sau rerank


# ============================================================
# Pipeline retrieval (lấy CẢ nội dung doc để judge)
# ============================================================

def _build_pipeline():
    from source.core.config import Settings
    from source.model.generate_model import Gemini
    from source.model.embedding_model import Sentences_Transformer_Embedding
    from source.model.rerank_model_finetune import RerankModelFinetune
    from source.data.vectordb.qdrant import Qdrant_Vector
    from source.search.utils_search import Qdrant_Utils
    from source.rerank.utils_rerank import Rerank_Utils
    from source.generate.generate import Gemini_Generate

    setting = Settings()
    gemini = Gemini(setting)
    embedding = Sentences_Transformer_Embedding(setting)
    qdrant = Qdrant_Vector(setting, embedding)
    rerank_utils = Rerank_Utils(RerankModelFinetune(setting))
    generate_utils = Gemini_Generate(gemini, setting)
    qdrant_utils = Qdrant_Utils(qdrant, generate_utils)
    return setting, generate_utils, qdrant_utils, rerank_utils


def _retrieve_top5(generate_utils, qdrant_utils, rerank_utils, query, entities, use_filter):
    """Trả về list top-5 sau rerank: [{content, sohieu, article}, ...]."""
    from pyvi import ViTokenizer

    queries = generate_utils.generate_query(query)
    metadata_filter = None
    if use_filter and entities:
        metadata_filter = qdrant_utils.build_metadata_filter(entities)

    query_results = []
    for q in queries:
        tokenized = ViTokenizer.tokenize(q)
        query_results.append(qdrant_utils.search_documents(tokenized, filter=metadata_filter))

    rrf = rerank_utils.reciprocal_rank_fusion(query_results)
    if not rrf:
        return []
    reranked = rerank_utils.rerank_documents_finetune(query, rrf)  # [(content, {score, doc_metadata})]
    out = []
    for content, info in reranked:
        md = info.get("doc_metadata", {}) or {}
        out.append({
            "content": (content or "").replace("_", " ").strip(),
            "sohieu": md.get("SoHieu", ""),
            "article": md.get("Article", ""),
        })
    return out


# ============================================================
# LLM judge
# ============================================================

JUDGE_SYSTEM = """\
Bạn là giám khảo đánh giá mức độ liên quan trong hệ thống truy xuất văn bản
pháp luật Việt Nam. Nhiệm vụ: cho một CÂU HỎI và một ĐOẠN VĂN BẢN LUẬT, hãy
xác định đoạn văn bản đó CÓ chứa thông tin trả lời (toàn bộ hoặc một phần
quan trọng) cho câu hỏi hay không.

Quy tắc:
- "relevant" = true nếu đoạn văn bản trực tiếp trả lời, hoặc chứa nội dung
  cốt lõi mà câu hỏi yêu cầu (kể cả khi không phải đúng điều luật người hỏi
  mong đợi, miễn là trả lời được câu hỏi).
- "relevant" = false nếu đoạn chỉ liên quan mơ hồ về chủ đề nhưng KHÔNG trả
  lời được câu hỏi, hoặc lạc đề.
- Chỉ trả về MỘT object JSON: {{"relevant": true/false, "reason": "ngắn gọn"}}.
- Không thêm giải thích ngoài JSON, không markdown fence."""

JUDGE_HUMAN = """\
CÂU HỎI:
{query}

ĐOẠN VĂN BẢN LUẬT:
{doc}

JSON:"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge(generate_utils, query: str, doc: str) -> bool:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    model = generate_utils._make_model(temperature=0.0, max_tokens=200)
    msgs = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM),
        ("human", JUDGE_HUMAN),
    ]).format_messages(query=query, doc=(doc or "")[:3000])
    raw = (model | StrOutputParser()).invoke(msgs).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    m = _JSON_RE.search(raw)
    if not m:
        return False
    try:
        return bool(json.loads(m.group(0)).get("relevant", False))
    except Exception:
        return False


# ============================================================
# 1) RUN — retrieval + judge, cache
# ============================================================

def run_predictions(limit=100, use_filter=True, resume=True):
    from tqdm import tqdm
    data = json.load(open(DATASET, encoding="utf-8"))
    if limit and limit > 0:
        data = data[:limit]

    done = {}
    if resume and CACHE.exists():
        for r in json.load(open(CACHE, encoding="utf-8")):
            if r.get("judged") is not None:
                done[r["id"]] = r
        print(f"[resume] đã có {len(done)} câu, sẽ bỏ qua.")

    setting, generate_utils, qdrant_utils, rerank_utils = _build_pipeline()

    rows = []
    for rec in tqdm(data, desc="Retrieval + LLM judge"):
        if rec["id"] in done:
            rows.append(done[rec["id"]])
            continue
        try:
            top5 = _retrieve_top5(generate_utils, qdrant_utils, rerank_utils,
                                  rec["query"], rec.get("entities"), use_filter)
            judged = []
            for d in top5:
                rel = _judge(generate_utils, rec["query"], d["content"])
                judged.append({"sohieu": d["sohieu"], "article": d["article"], "relevant": rel})
        except Exception as e:
            print(f"\n[!] lỗi: {rec['id']} :: {type(e).__name__}: {e}", flush=True)
            judged = []
        rows.append({
            "id": rec["id"], "domain": rec["domain"], "query": rec["query"],
            "has_entities": rec.get("has_entities"),
            "gold_sohieu": rec["gold_sohieu"], "gold_article": rec["gold_article"],
            "judged": judged,
        })
        json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(rows, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu {len(rows)} kết quả vào {CACHE.name}")
    return rows


# ============================================================
# 2) SCORE — metric theo phán quyết judge
# ============================================================

def _first_relevant_rank(judged):
    for i, j in enumerate(judged):
        if j.get("relevant"):
            return i + 1
    return 0


def _agg(rows):
    n = len(rows)
    if n == 0:
        return {}
    out = {}
    for k in (1, 3, 5):
        hits = sum(1 for r in rows
                   if any(j.get("relevant") for j in r["judged"][:k]))
        out[f"hit@{k}"] = hits / n
    mrr = ndcg = prec = 0.0
    for r in rows:
        judged = r["judged"][:5]
        rank = _first_relevant_rank(judged)
        if rank:
            mrr += 1.0 / rank
            # NDCG nhị phân: DCG theo các vị trí relevant; IDCG = tài liệu
            # relevant xếp lên đầu.
            rels = [1 if j.get("relevant") else 0 for j in judged]
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
            n_rel = sum(rels)
            idcg = sum(1 / math.log2(i + 2) for i in range(n_rel)) if n_rel else 0.0
            ndcg += (dcg / idcg) if idcg else 0.0
        prec += sum(1 for j in judged if j.get("relevant")) / 5.0
    out["mrr@5"] = mrr / n
    out["ndcg@5"] = ndcg / n
    out["precision@5"] = prec / n
    out["n"] = n
    return out


def score(rows):
    rows = [r for r in rows if r.get("judged") is not None]
    report = {"overall": _agg(rows), "by_domain": {}, "by_has_entities": {}}
    for d in sorted({r["domain"] for r in rows}):
        report["by_domain"][d] = _agg([r for r in rows if r["domain"] == d])
    for he in (True, False):
        sub = [r for r in rows if r.get("has_entities") == he]
        if sub:
            report["by_has_entities"]["with_entities" if he else "no_entities"] = _agg(sub)
    return report


def print_report(rep):
    line = "=" * 64
    o = rep["overall"]
    print("\n" + line)
    print("ĐÁNH GIÁ RETRIEVAL — LLM-as-JUDGE (top-5 sau rerank)")
    print(line)
    print(f"Số câu        : {o.get('n', 0)}")
    print(f"  Hit@1       : {o.get('hit@1', 0):.4f}")
    print(f"  Hit@3       : {o.get('hit@3', 0):.4f}")
    print(f"  Hit@5       : {o.get('hit@5', 0):.4f}")
    print(f"  MRR@5       : {o.get('mrr@5', 0):.4f}")
    print(f"  NDCG@5      : {o.get('ndcg@5', 0):.4f}")
    print(f"  Precision@5 : {o.get('precision@5', 0):.4f}")
    print("\n[THEO DOMAIN]  (Hit@5 / MRR@5 / Prec@5)")
    for d, m in rep["by_domain"].items():
        print(f"  {d:<22}{m.get('hit@5',0):>8.4f}{m.get('mrr@5',0):>10.4f}{m.get('precision@5',0):>10.4f}  (n={m.get('n',0)})")
    print("\n[THEO LOẠI CÂU]  (Hit@5 / Prec@5)")
    for d, m in rep["by_has_entities"].items():
        print(f"  {d:<16}{m.get('hit@5',0):>8.4f}{m.get('precision@5',0):>10.4f}  (n={m.get('n',0)})")
    print(line)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Eval retrieval bằng LLM-as-judge")
    ap.add_argument("mode", choices=["run", "score", "all"], nargs="?", default="all")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.mode in ("run", "all"):
        run_predictions(limit=args.limit, use_filter=not args.no_filter,
                        resume=not args.no_resume)

    if args.mode in ("score", "all"):
        if not CACHE.exists():
            print(f"Chưa có cache {CACHE.name}. Chạy `run` trước.")
            sys.exit(1)
        rows = json.load(open(CACHE, encoding="utf-8"))
        rep = score(rows)
        json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print_report(rep)
        print(f"✓ Đã lưu report vào {REPORT.name}")


if __name__ == "__main__":
    main()
