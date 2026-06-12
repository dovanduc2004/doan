"""Sinh bộ dữ liệu đánh giá RETRIEVAL (~200 câu) — Cách 1: back-generation.

Ý tưởng (đáng tin vì gold xác định theo cấu trúc, không đoán):
  - Lấy từng chunk CÓ THẬT từ corpus đã embed vào Qdrant
    (dataset/*/handled_semantic_Chunk_content_*_Extracted.json).
  - Sinh câu hỏi TỪ chunk đó bằng template (dựa trên tiêu đề Điều luật).
    => Đáp án của câu hỏi nằm CHÍNH trong chunk đó.
  - Gold = (SoHieu, Article) của chunk nguồn → khớp thẳng, không gán tay.

Mỗi chunk sinh 1 trong 2 kiểu câu:
  - has_entities=True : câu nhắc rõ văn bản (Loại + Số hiệu) → test cả metadata filter.
  - has_entities=False: câu chỉ hỏi chủ đề → test thuần embedding + rerank.

Chạy:
    python build_retrieval_dataset.py                 # mặc định ~200 câu
    python build_retrieval_dataset.py --target 200
"""
import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT = HERE / "retrieval_eval_dataset.json"
SEED = 20240601

# Bỏ các điều luật "khung" không mang nội dung trả lời cụ thể.
BOILERPLATE = re.compile(
    r"(thi hành|hiệu lực|phạm vi điều chỉnh|điều khoản chuyển tiếp|giải thích từ ngữ|"
    r"đối tượng áp dụng|trách nhiệm thi hành|quy định chuyển tiếp|tổ chức thực hiện|"
    r"điều khoản thi hành|khen thưởng|xử lý vi phạm)",
    re.I,
)


def lower_first(s):
    return s[0].lower() + s[1:] if s else s


def load_corpus():
    """Trả về list chunk sạch, mỗi chunk gồm metadata + topic (tiêu đề Điều)."""
    files = sorted(glob.glob(str(REPO_ROOT / "dataset" / "*" /
                                 "handled_semantic_Chunk_content_*_Extracted.json")))
    seen = set()
    items = []
    for f in files:
        domain = Path(f).parent.name
        data = json.load(open(f, encoding="utf-8"))
        for it in data:
            art = (it.get("Article") or "").strip()
            sh = it.get("SoHieu")
            lv = it.get("LoaiVanBan")
            nb = it.get("NoiBanHanh")
            if not (art and sh and lv and nb):
                continue
            m = re.match(r"^Điều\s+\d+[\.:]?\s*(.+)$", art)
            if not m:
                continue
            topic = m.group(1).strip().rstrip(".")
            if len(topic) < 8 or BOILERPLATE.search(topic):
                continue
            key = (sh, art)
            if key in seen:
                continue
            seen.add(key)
            nf = it.get("NgayBanHanhFilter") or []
            year = nf[-1] if nf else None
            items.append({
                "domain": domain,
                "LoaiVanBan": lv,
                "SoHieu": sh,
                "NoiBanHanh": nb,
                "Year": str(year) if year else None,
                "Article": art,
                "topic": topic,
                "content": (it.get("OriginalContent") or "")[:600],
            })
    return items


# Câu hỏi CÓ nhắc văn bản (3 entity gold: loại + nơi ban hành + năm)
ENT_Q_TEMPLATES = [
    "{lv} {sh} quy định về {topic} như thế nào?",
    "Theo {lv} {sh} của {nb}, {topic_lc} được quy định ra sao?",
    "{lv} {sh} có quy định gì về {topic}?",
    "Nội dung về {topic} trong {lv} {sh} là gì?",
]

# Câu hỏi CHỈ hỏi chủ đề (không nhắc văn bản)
NOENT_Q_TEMPLATES = [
    "{Topic} được pháp luật quy định như thế nào?",
    "Quy định về {topic} ra sao?",
    "Pháp luật Việt Nam quy định thế nào về {topic}?",
    "{Topic} có những quy định gì?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(SEED)

    corpus = load_corpus()
    print(f"Corpus sạch: {len(corpus)} (doc, article) pairs")

    # Cân bằng theo domain bằng round-robin.
    by_dom = defaultdict(list)
    for it in corpus:
        by_dom[it["domain"]].append(it)
    doms = sorted(by_dom)
    for d in doms:
        rng.shuffle(by_dom[d])

    picks, idx = [], {d: 0 for d in doms}
    while len(picks) < args.target:
        progressed = False
        for d in doms:
            if len(picks) >= args.target:
                break
            if idx[d] < len(by_dom[d]):
                picks.append(by_dom[d][idx[d]])
                idx[d] += 1
                progressed = True
        if not progressed:
            break

    rng.shuffle(picks)

    records = []
    counts = defaultdict(int)
    for i, it in enumerate(picks):
        lv, sh, nb, topic = it["LoaiVanBan"], it["SoHieu"], it["NoiBanHanh"], it["topic"]
        # Xen kẽ: chẵn → có entity, lẻ → không entity
        has_ent = (i % 2 == 0)
        if has_ent:
            qt = ENT_Q_TEMPLATES[i % len(ENT_Q_TEMPLATES)]
            q = qt.format(lv=lv, sh=sh, nb=nb, topic=topic, topic_lc=lower_first(topic))
            entities = {"LoaiVanBanFilter": it.get("LoaiVanBan", "").lower(),
                        "NoibanHanhFilter": it.get("NoiBanHanh", "").lower()}
            if it["Year"]:
                entities["NgayBanHanhFilter"] = it["Year"]
            prefix = "ret_ent"
        else:
            qt = NOENT_Q_TEMPLATES[i % len(NOENT_Q_TEMPLATES)]
            q = qt.format(topic=lower_first(topic), Topic=topic)
            entities = None
            prefix = "ret_noent"
        counts[prefix] += 1
        records.append({
            "id": f"{prefix}_{counts[prefix]:03d}",
            "domain": it["domain"],
            "query": q,
            "gold_sohieu": sh,
            "gold_article": it["Article"],
            "entities": entities,
            "has_entities": has_ent,
        })

    rng.shuffle(records)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Tổng số mẫu: {len(records)} → {OUT.name}")
    print("Theo domain :", dict(Counter(r["domain"] for r in records)))
    print("Có entity   :", dict(Counter(r["has_entities"] for r in records)))


if __name__ == "__main__":
    main()
