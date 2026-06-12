"""Planner component evaluation.

Đánh giá PlannerAgent của hệ thống Agentic RAG
hiện tại.

Planner gộp nhiều "đầu ra con" vào 1 LLM call → mỗi đầu ra được chấm bằng
metric phù hợp với bản chất của nó:

  - intent              : phân loại 4 lớp  → accuracy, macro-F1, P/R/F1 theo lớp, confusion matrix
  - tool                : phân loại 5 lớp  → accuracy tổng + tách theo web_search_enabled
  - entities            : trích xuất field → precision/recall/F1 mức field + exact-set-match
  - needs_clarification : nhị phân         → precision/recall/F1
  - rewritten_query     : sinh văn bản      → (tùy chọn) cosine similarity với gold rewrite
  - robustness          : vận hành          → fallback rate, latency p50/p95

Quy trình (tránh tốn tiền Bedrock khi rerun metric):
  1) `run`  : gọi planner trên từng mẫu → lưu raw output ra file cache JSON.
  2) `score`: đọc cache → tính toàn bộ metric → in báo cáo + lưu report JSON.
  3) `all`  : chạy cả hai.

Cách dùng:
    # Chạy planner và tính metric (cần AWS credentials cho Bedrock)
    python eval_planner.py all

    # Chỉ chạy lại metric từ cache đã có (không gọi LLM)
    python eval_planner.py score

    # Chạy mỗi mẫu nhiều lần để đo độ ổn định (consistency) ở temperature=0
    python eval_planner.py run --repeat 3

    # Bật đo cosine similarity cho rewritten_query (load model embedding, nặng hơn)
    python eval_planner.py score --rewrite-sim
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

# --- Cho phép import package `source.*` từ repo root ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]  # eval -> Agentic_RAG -> source_research -> <root>
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_PATH = THIS_DIR / "planner_eval_dataset.json"
CACHE_PATH = THIS_DIR / "planner_eval_predictions.json"
REPORT_PATH = THIS_DIR / "planner_eval_report.json"

INTENT_CLASSES = ["chitchat", "legal", "realtime", "out_of_scope"]
TOOL_CLASSES = ["template", "rag", "web", "reject", "ask_user"]
ENTITY_FIELDS = ["LoaiVanBanFilter", "NoibanHanhFilter", "NgayBanHanhFilter"]
FALLBACK_REASON_MARKER = "không parse được"  # xuất hiện trong _fallback_plan.reason


# ============================================================
# 1) RUN — gọi planner, cache kết quả
# ============================================================

def _build_planner():
    """Khởi tạo PlannerAgent với generator thật (Bedrock/Gemini theo .env).

    Chỉ load đúng thứ cần cho planner (Settings + Gemini + Gemini_Generate),
    KHÔNG load embedding/BERT/CrossEncoder → nhẹ và nhanh.
    """
    from source.core.config import Settings
    from source.model.generate_model import Gemini
    from source.generate.generate import Gemini_Generate
    from source.agent.planner import PlannerAgent

    settings = Settings()
    gemini = Gemini(settings)
    generator = Gemini_Generate(gemini, settings)
    return PlannerAgent(generator)


def load_dataset() -> List[Dict[str, Any]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_predictions(repeat: int = 1) -> List[Dict[str, Any]]:
    """Chạy planner trên toàn bộ dataset, lưu raw output + latency ra cache."""
    planner = _build_planner()
    dataset = load_dataset()
    records: List[Dict[str, Any]] = []

    for i, sample in enumerate(dataset, 1):
        ctx = sample.get("conversation_context", "") or ""
        query = sample["user_query"]
        web = bool(sample["web_search_enabled"])

        runs: List[Dict[str, Any]] = []
        for r in range(repeat):
            t0 = time.time()
            try:
                plan = planner.plan(query, ctx, web)
                elapsed_ms = int((time.time() - t0) * 1000)
                runs.append({
                    "intent": plan.intent,
                    "tool": plan.tool,
                    "rewritten_query": plan.rewritten_query,
                    "entities": plan.entities,
                    "needs_clarification": plan.needs_clarification,
                    "reason": plan.reason,
                    "elapsed_ms": elapsed_ms,
                    "error": None,
                })
            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                runs.append({
                    "intent": None, "tool": None, "rewritten_query": None,
                    "entities": None, "needs_clarification": None, "reason": None,
                    "elapsed_ms": elapsed_ms, "error": f"{type(e).__name__}: {e}",
                })

        # Lượt đầu là prediction chính dùng để chấm; runs giữ lại để đo consistency.
        primary = runs[0]
        records.append({
            "id": sample["id"],
            "user_query": query,
            "web_search_enabled": web,
            "prediction": primary,
            "runs": runs,
        })
        status = primary.get("error") or f"{primary['intent']}/{primary['tool']}"
        print(f"[{i}/{len(dataset)}] {sample['id']}: {status} ({primary['elapsed_ms']}ms)", flush=True)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu {len(records)} prediction vào {CACHE_PATH.name}")
    return records


# ============================================================
# 2) SCORE — tính metric từ cache
# ============================================================

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _normalize_entities(ent: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Chuẩn hóa entity dict về {field: value_lowercase}, bỏ field rỗng.

    Đồng bộ với Qdrant_Utils.build_metadata_filter (so khớp lowercase).
    """
    if not isinstance(ent, dict):
        return {}
    out = {}
    for k, v in ent.items():
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            out[k] = sv.lower()
    return out


def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _classification_metrics(pairs, classes):
    """pairs: list[(gold, pred)]. Trả accuracy, macro-F1, per-class, confusion."""
    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    accuracy = correct / n if n else 0.0

    per_class = {}
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        support = sum(1 for g, _ in pairs if g == c)
        pr, rc, f1 = _prf(tp, fp, fn)
        per_class[c] = {"precision": pr, "recall": rc, "f1": f1, "support": support}
        if support > 0:  # macro-F1 chỉ tính trên lớp có mẫu
            f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Confusion matrix: confusion[gold][pred]
    confusion = {g: {p: 0 for p in classes + ["<other>"]} for g in classes}
    for g, p in pairs:
        if g not in confusion:
            continue
        col = p if p in classes else "<other>"
        confusion[g][col] += 1

    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def score(records: List[Dict[str, Any]], dataset: List[Dict[str, Any]],
          rewrite_sim: bool = False) -> Dict[str, Any]:
    by_id = {s["id"]: s for s in dataset}

    intent_pairs = []           # (gold_primary, pred)
    intent_correct_accept = 0   # đúng nếu pred ∈ acceptable_intents
    tool_pairs = []
    tool_correct_accept = 0
    tool_pairs_by_web = {True: [], False: []}

    # entities (chỉ trên mẫu có expected_entities hoặc pred có entities)
    ent_tp = ent_fp = ent_fn = 0
    ent_exact_total = 0
    ent_exact_match = 0

    # needs_clarification (nhị phân)
    clar_tp = clar_fp = clar_fn = clar_tn = 0

    # robustness
    n = len(records)
    n_error = 0
    n_fallback = 0
    latencies = []

    # consistency
    consistency_scores = []

    # legal routing accuracy theo domain thật (chỉ trên mẫu legal)
    domain_total = defaultdict(int)
    domain_intent_ok = defaultdict(int)

    # rewrite similarity (tùy chọn)
    rewrite_sims = []
    embed_fn = None
    if rewrite_sim:
        embed_fn = _make_embedder()

    per_sample_rows = []

    for rec in records:
        sid = rec["id"]
        gold = by_id.get(sid)
        if gold is None:
            continue
        pred = rec["prediction"]
        web = rec["web_search_enabled"]

        latencies.append(pred.get("elapsed_ms", 0))
        if pred.get("error"):
            n_error += 1
        if pred.get("reason") and FALLBACK_REASON_MARKER in pred["reason"].lower():
            n_fallback += 1

        # ----- intent -----
        gi = gold["expected_intent"]
        pi = pred.get("intent")
        accept_intents = gold.get("acceptable_intents") or [gi]
        intent_pairs.append((gi, pi if pi in INTENT_CLASSES else "<other>"))
        intent_ok = pi in accept_intents
        if intent_ok:
            intent_correct_accept += 1

        # ----- legal accuracy theo domain -----
        dom = gold.get("domain") or ""
        if gi == "legal" and dom:
            domain_total[dom] += 1
            if intent_ok and (pred.get("tool") in (gold.get("acceptable_tools") or [gold["expected_tool"]])):
                domain_intent_ok[dom] += 1

        # ----- tool -----
        gt = gold["expected_tool"]
        pt = pred.get("tool")
        accept_tools = gold.get("acceptable_tools") or [gt]
        tool_pairs.append((gt, pt if pt in TOOL_CLASSES else "<other>"))
        tool_pairs_by_web[web].append((gt, pt if pt in TOOL_CLASSES else "<other>"))
        tool_ok = pt in accept_tools
        if tool_ok:
            tool_correct_accept += 1

        # ----- entities -----
        gold_ent = _normalize_entities(gold.get("expected_entities"))
        pred_ent = _normalize_entities(pred.get("entities"))
        # chỉ tính exact-set-match khi có ít nhất một bên có entity
        if gold_ent or pred_ent:
            ent_exact_total += 1
            if gold_ent == pred_ent:
                ent_exact_match += 1
        for field in ENTITY_FIELDS:
            g_has = field in gold_ent
            p_has = field in pred_ent
            if g_has and p_has:
                if gold_ent[field] == pred_ent[field]:
                    ent_tp += 1
                else:
                    # giá trị sai = vừa fp (đoán sai) vừa fn (bỏ lỡ đúng)
                    ent_fp += 1
                    ent_fn += 1
            elif p_has and not g_has:
                ent_fp += 1
            elif g_has and not p_has:
                ent_fn += 1

        # ----- needs_clarification -----
        gc = bool(gold.get("expected_needs_clarification"))
        pc = bool(pred.get("needs_clarification"))
        if gc and pc:
            clar_tp += 1
        elif pc and not gc:
            clar_fp += 1
        elif gc and not pc:
            clar_fn += 1
        else:
            clar_tn += 1

        # ----- consistency (nếu có nhiều run) -----
        runs = rec.get("runs", [])
        if len(runs) > 1:
            sigs = [f"{r.get('intent')}|{r.get('tool')}" for r in runs]
            most = Counter(sigs).most_common(1)[0][1]
            consistency_scores.append(most / len(sigs))

        # ----- rewrite similarity -----
        sim = None
        if embed_fn is not None:
            gq = gold.get("expected_rewritten_query") or ""
            pq = pred.get("rewritten_query") or ""
            if gq and pq:
                sim = embed_fn(gq, pq)
                rewrite_sims.append(sim)

        per_sample_rows.append({
            "id": sid,
            "web": web,
            "intent_gold": gi, "intent_pred": pi, "intent_ok": intent_ok,
            "tool_gold": gt, "tool_pred": pt, "tool_ok": tool_ok,
            "entities_gold": gold_ent or None, "entities_pred": pred_ent or None,
            "needs_clar_gold": gc, "needs_clar_pred": pc,
            "rewrite_sim": sim,
            "rewritten_query_pred": pred.get("rewritten_query"),
        })

    # ---- tổng hợp ----
    intent_metrics = _classification_metrics(intent_pairs, INTENT_CLASSES)
    intent_metrics["accuracy_with_acceptable"] = intent_correct_accept / n if n else 0.0

    tool_metrics = _classification_metrics(tool_pairs, TOOL_CLASSES)
    tool_metrics["accuracy_with_acceptable"] = tool_correct_accept / n if n else 0.0
    tool_metrics["accuracy_web_on"] = _classification_metrics(
        tool_pairs_by_web[True], TOOL_CLASSES)["accuracy"] if tool_pairs_by_web[True] else None
    tool_metrics["accuracy_web_off"] = _classification_metrics(
        tool_pairs_by_web[False], TOOL_CLASSES)["accuracy"] if tool_pairs_by_web[False] else None

    ent_pr, ent_rc, ent_f1 = _prf(ent_tp, ent_fp, ent_fn)
    entity_metrics = {
        "field_precision": ent_pr,
        "field_recall": ent_rc,
        "field_f1": ent_f1,
        "tp": ent_tp, "fp": ent_fp, "fn": ent_fn,
        "exact_set_match": (ent_exact_match / ent_exact_total) if ent_exact_total else None,
        "exact_set_total": ent_exact_total,
    }

    clar_pr, clar_rc, clar_f1 = _prf(clar_tp, clar_fp, clar_fn)
    clarification_metrics = {
        "precision": clar_pr, "recall": clar_rc, "f1": clar_f1,
        "tp": clar_tp, "fp": clar_fp, "fn": clar_fn, "tn": clar_tn,
    }

    lat_sorted = sorted(latencies) if latencies else [0]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))]
    robustness = {
        "n_samples": n,
        "error_rate": n_error / n if n else 0.0,
        "fallback_rate": n_fallback / n if n else 0.0,
        "json_parse_success_rate": (n - n_error - n_fallback) / n if n else 0.0,
        "latency_ms_p50": median(latencies) if latencies else 0,
        "latency_ms_p95": p95,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0,
        "consistency_at_t0": (sum(consistency_scores) / len(consistency_scores))
                              if consistency_scores else None,
    }

    rewrite_metrics = {
        "mean_cosine_sim": (sum(rewrite_sims) / len(rewrite_sims)) if rewrite_sims else None,
        "n_compared": len(rewrite_sims),
        "note": "Bật bằng --rewrite-sim. Khuyến nghị bổ sung LLM-as-judge cho đánh giá đầy đủ.",
    }

    domain_breakdown = {
        d: {"accuracy": domain_intent_ok[d] / domain_total[d], "n": domain_total[d]}
        for d in sorted(domain_total)
    }

    return {
        "intent": intent_metrics,
        "tool": tool_metrics,
        "entities": entity_metrics,
        "needs_clarification": clarification_metrics,
        "rewritten_query": rewrite_metrics,
        "robustness": robustness,
        "legal_by_domain": domain_breakdown,
        "per_sample": per_sample_rows,
    }


def _make_embedder():
    """Trả về hàm cosine(a, b) dùng model embedding của hệ thống."""
    from source.core.config import Settings
    from source.model.embedding_model import Sentences_Transformer_Embedding
    from sentence_transformers import util

    settings = Settings()
    emb = Sentences_Transformer_Embedding(settings)

    def cosine(a: str, b: str) -> float:
        va = emb.embeddings_bkai.embed_query(a)
        vb = emb.embeddings_bkai.embed_query(b)
        return float(util.cos_sim(va, vb)[0][0])

    return cosine


# ============================================================
# 3) In báo cáo
# ============================================================

def _fmt_pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "N/A"


def print_report(report: Dict[str, Any]):
    line = "=" * 64
    print("\n" + line)
    print("BÁO CÁO ĐÁNH GIÁ PLANNER (component-level)")
    print(line)

    im = report["intent"]
    print(f"\n[1] INTENT (n={im['n']})")
    print(f"    Accuracy (strict)          : {_fmt_pct(im['accuracy'])}")
    print(f"    Accuracy (acceptable set)  : {_fmt_pct(im['accuracy_with_acceptable'])}")
    print(f"    Macro-F1                   : {_fmt_pct(im['macro_f1'])}")
    print("    Theo lớp:")
    for c, m in im["per_class"].items():
        print(f"      {c:<14} P={_fmt_pct(m['precision'])}  R={_fmt_pct(m['recall'])}  "
              f"F1={_fmt_pct(m['f1'])}  (support={m['support']})")
    print("    Confusion matrix (hàng=gold, cột=pred):")
    cols = INTENT_CLASSES + ["<other>"]
    print("      " + " " * 14 + "".join(f"{c[:8]:>10}" for c in cols))
    for g in INTENT_CLASSES:
        row = im["confusion"][g]
        print(f"      {g:<14}" + "".join(f"{row[c]:>10}" for c in cols))

    tm = report["tool"]
    print(f"\n[2] TOOL (n={tm['n']})")
    print(f"    Accuracy (strict)          : {_fmt_pct(tm['accuracy'])}")
    print(f"    Accuracy (acceptable set)  : {_fmt_pct(tm['accuracy_with_acceptable'])}")
    print(f"    Accuracy (web ON)          : {_fmt_pct(tm['accuracy_web_on'])}")
    print(f"    Accuracy (web OFF)         : {_fmt_pct(tm['accuracy_web_off'])}")
    print(f"    Macro-F1                   : {_fmt_pct(tm['macro_f1'])}")
    for c, m in tm["per_class"].items():
        print(f"      {c:<14} P={_fmt_pct(m['precision'])}  R={_fmt_pct(m['recall'])}  "
              f"F1={_fmt_pct(m['f1'])}  (support={m['support']})")

    em = report["entities"]
    print("\n[3] ENTITIES (mức field)")
    print(f"    Precision                  : {_fmt_pct(em['field_precision'])}")
    print(f"    Recall                     : {_fmt_pct(em['field_recall'])}")
    print(f"    F1                         : {_fmt_pct(em['field_f1'])}")
    print(f"    Exact-set-match            : {_fmt_pct(em['exact_set_match'])} "
          f"(trên {em['exact_set_total']} mẫu có entity)")
    print(f"    tp={em['tp']} fp={em['fp']} fn={em['fn']}")

    cm = report["needs_clarification"]
    print("\n[4] NEEDS_CLARIFICATION (nhị phân)")
    print(f"    Precision={_fmt_pct(cm['precision'])}  Recall={_fmt_pct(cm['recall'])}  "
          f"F1={_fmt_pct(cm['f1'])}")
    print(f"    tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}")

    rw = report["rewritten_query"]
    print("\n[5] REWRITTEN_QUERY")
    if rw["mean_cosine_sim"] is not None:
        print(f"    Mean cosine sim với gold   : {rw['mean_cosine_sim']:.4f} "
              f"(n={rw['n_compared']})")
    else:
        print("    (Chưa đo — chạy với --rewrite-sim để bật)")

    dbd = report.get("legal_by_domain") or {}
    if dbd:
        print("\n[5b] LEGAL ROUTING ACCURACY THEO DOMAIN (intent+tool đúng)")
        for d, m in dbd.items():
            print(f"      {d:<20} {_fmt_pct(m['accuracy'])}  (n={m['n']})")

    rb = report["robustness"]
    print("\n[6] ROBUSTNESS / VẬN HÀNH")
    print(f"    JSON parse success rate    : {_fmt_pct(rb['json_parse_success_rate'])}")
    print(f"    Fallback rate              : {_fmt_pct(rb['fallback_rate'])}")
    print(f"    Error rate                 : {_fmt_pct(rb['error_rate'])}")
    print(f"    Latency p50 / p95 / mean   : {rb['latency_ms_p50']:.0f} / "
          f"{rb['latency_ms_p95']:.0f} / {rb['latency_ms_mean']:.0f} ms")
    if rb["consistency_at_t0"] is not None:
        print(f"    Consistency @ T=0          : {_fmt_pct(rb['consistency_at_t0'])}")

    # Liệt kê các mẫu sai để soi nhanh
    wrong = [r for r in report["per_sample"] if not r["intent_ok"] or not r["tool_ok"]]
    if wrong:
        print(f"\n[!] {len(wrong)} mẫu sai intent/tool:")
        for r in wrong:
            tag = []
            if not r["intent_ok"]:
                tag.append(f"intent: {r['intent_gold']}→{r['intent_pred']}")
            if not r["tool_ok"]:
                tag.append(f"tool: {r['tool_gold']}→{r['tool_pred']}")
            print(f"      {r['id']:<22} (web={r['web']}) " + " ; ".join(tag))
    print("\n" + line)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Đánh giá PlannerAgent (component-level)")
    parser.add_argument("mode", choices=["run", "score", "all"], default="all", nargs="?",
                        help="run = gọi LLM & cache; score = tính metric từ cache; all = cả hai")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Số lần chạy mỗi mẫu (đo consistency). Mặc định 1.")
    parser.add_argument("--rewrite-sim", action="store_true",
                        help="Bật đo cosine similarity cho rewritten_query (load model embedding).")
    args = parser.parse_args()

    if args.mode in ("run", "all"):
        run_predictions(repeat=args.repeat)

    if args.mode in ("score", "all"):
        if not CACHE_PATH.exists():
            print(f"Chưa có cache {CACHE_PATH.name}. Chạy `python eval_planner.py run` trước.")
            sys.exit(1)
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        dataset = load_dataset()
        report = score(records, dataset, rewrite_sim=args.rewrite_sim)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print_report(report)
        print(f"✓ Đã lưu report chi tiết vào {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
