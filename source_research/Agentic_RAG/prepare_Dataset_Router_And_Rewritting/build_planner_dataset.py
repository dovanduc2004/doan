"""Sinh bộ dữ liệu đánh giá Planner (~1000 mẫu) — GROUNDED trên dữ liệu thật.

Chiến lược scale lên 1000 mà vẫn giữ nhãn chính xác:
  - Phần `legal` (đa số) được MINE TRỰC TIẾP từ corpus đã embed vào Qdrant
    (các file handled_semantic_Chunk_content_*_Extracted.json của 5 domain).
    Mỗi câu hỏi được dựng từ một (văn bản, Điều luật) CÓ THẬT; nhãn entity
    (LoaiVanBanFilter / NoibanHanhFilter / NgayBanHanhFilter) được SUY THẲNG
    từ metadata của chính chunk đó → gold luôn khớp CSDL, không gán tay.
  - `tool` suy ra tự động theo bảng định tuyến PlannerAgent._enforce_routing.
  - chitchat / realtime / out_of_scope: pool curate có kiểm soát, nhân bản
    có biến thể (web on/off, paraphrase) để đủ số lượng.

Đánh đổi: câu legal sinh theo template nên "máy móc" hơn câu viết tay, nhưng
nội dung/entity bám sát dữ liệu thực → phản ánh đúng phân phối user sẽ hỏi.

Chạy:
    python build_planner_dataset.py                 # mặc định ~1000 mẫu
    python build_planner_dataset.py --target 1000   # chỉnh số lượng
"""
import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT = HERE / "planner_eval_dataset.json"
SEED = 20240601

BOILERPLATE = re.compile(
    r"(thi hành|hiệu lực|phạm vi điều chỉnh|điều khoản chuyển tiếp|giải thích từ ngữ|"
    r"đối tượng áp dụng|trách nhiệm thi hành|quy định chuyển tiếp|tổ chức thực hiện|"
    r"điều khoản thi hành|khen thưởng|xử lý vi phạm)",
    re.I,
)

# Bộ Luật/Luật → Quốc hội, Nghị định → Chính phủ (suy ra issuer chuẩn).
ISSUER_INFERABLE = {"luật", "nghị định"}


def derive_tool(intent, web, needs_clar):
    if intent == "chitchat":
        return "template"
    if intent == "legal":
        return "rag"
    if not web:
        return "reject"
    return "ask_user" if needs_clar else "web"


records = []
_counts = {}


def add(prefix, intent, query, web, rewrite, ctx="", entities=None,
        needs_clar=False, acc_intents=None, acc_tools=None, domain="", note=""):
    _counts[prefix] = _counts.get(prefix, 0) + 1
    rec = {
        "id": f"{prefix}_{_counts[prefix]:03d}",
        "domain": domain,
        "conversation_context": ctx,
        "user_query": query,
        "web_search_enabled": web,
        "expected_intent": intent,
        "expected_tool": derive_tool(intent, web, needs_clar),
        "expected_rewritten_query": rewrite,
        "expected_entities": entities,
        "expected_needs_clarification": needs_clar,
        "note": note,
    }
    if acc_intents:
        rec["acceptable_intents"] = acc_intents
    if acc_tools:
        rec["acceptable_tools"] = acc_tools
    records.append(rec)


# =====================================================================
# Mine corpus thật
# =====================================================================
def load_corpus():
    """Trả về list các (doc,article) sạch: dict gồm metadata cần thiết."""
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
                "topic": topic,
            })
    return items


def lower_first(s):
    return s[0].lower() + s[1:] if s else s


# Template câu hỏi LEGAL CÓ entity (nhắc rõ văn bản → 3 entity gold)
ENT_Q_TEMPLATES = [
    "{lv} {sh} quy định về {topic} như thế nào?",
    "Theo {lv} {sh} của {nb}, {topic_lc} được quy định ra sao?",
    "{lv} {sh} có quy định gì về {topic}?",
    "Nội dung về {topic} trong {lv} {sh} là gì?",
    "{nb} quy định về {topic} tại {lv} {sh} thế nào?",
]
ENT_RW_TEMPLATES = [
    "{lv} {sh} của {nb} quy định về {topic} như thế nào?",
    "Theo {lv} {sh} do {nb} ban hành, quy định về {topic} ra sao?",
    "{lv} {sh} của {nb} có những quy định gì về {topic}?",
]

# Template câu hỏi LEGAL KHÔNG entity (chỉ chủ đề, không nhắc văn bản)
NOENT_Q_TEMPLATES = [
    "{Topic} được pháp luật quy định như thế nào?",
    "Quy định về {topic} ra sao?",
    "Pháp luật Việt Nam quy định thế nào về {topic}?",
    "{Topic} có những quy định gì?",
    "Cho tôi hỏi về quy định {topic}?",
]
NOENT_RW_TEMPLATES = [
    "Quy định của pháp luật Việt Nam về {topic} như thế nào?",
    "Pháp luật Việt Nam quy định về {topic} ra sao?",
    "Các quy định liên quan đến {topic} theo pháp luật hiện hành là gì?",
]


def build_legal(corpus, n_ent, n_noent, rng):
    rng.shuffle(corpus)
    # cân bằng domain: round-robin theo domain
    by_dom = defaultdict(list)
    for it in corpus:
        by_dom[it["domain"]].append(it)
    doms = sorted(by_dom)
    for d in doms:
        rng.shuffle(by_dom[d])

    def round_robin(n):
        out, idx = [], {d: 0 for d in doms}
        while len(out) < n:
            progressed = False
            for d in doms:
                if idx[d] < len(by_dom[d]):
                    out.append(by_dom[d][idx[d]])
                    idx[d] += 1
                    progressed = True
                    if len(out) >= n:
                        break
            if not progressed:
                break
        return out

    picks = round_robin(n_ent + n_noent)
    rng.shuffle(picks)
    ent_picks = picks[:n_ent]
    noent_picks = picks[n_ent:n_ent + n_noent]

    for i, it in enumerate(ent_picks):
        lv, sh, nb, topic = it["LoaiVanBan"], it["SoHieu"], it["NoiBanHanh"], it["topic"]
        qt = ENT_Q_TEMPLATES[i % len(ENT_Q_TEMPLATES)]
        rt = ENT_RW_TEMPLATES[i % len(ENT_RW_TEMPLATES)]
        q = qt.format(lv=lv, sh=sh, nb=nb, topic=topic, topic_lc=lower_first(topic))
        rw = rt.format(lv=lv, sh=sh, nb=nb, topic=topic)
        ent = {"LoaiVanBanFilter": lv, "NoibanHanhFilter": nb}
        if it["Year"]:
            ent["NgayBanHanhFilter"] = it["Year"]
        web = (i % 6 == 0)
        add("legal_ent", "legal", q, web, rw, entities=ent, domain=it["domain"],
            note="Legal có văn bản cụ thể (mine từ corpus; entity suy từ metadata).")

    for i, it in enumerate(noent_picks):
        topic = it["topic"]
        qt = NOENT_Q_TEMPLATES[i % len(NOENT_Q_TEMPLATES)]
        rt = NOENT_RW_TEMPLATES[i % len(NOENT_RW_TEMPLATES)]
        q = qt.format(topic=lower_first(topic), Topic=topic)
        rw = rt.format(topic=lower_first(topic))
        web = (i % 5 == 0)
        add("legal_noent", "legal", q, web, rw, domain=it["domain"],
            note="Legal không văn bản cụ thể (mine chủ đề từ corpus).")


# =====================================================================
# Non-legal pools (curate + nhân bản có biến thể)
# =====================================================================
CHITCHAT = [
    "Xin chào, bạn khỏe không?", "Chào buổi sáng nhé!", "hello bạn ơi", "Bạn là ai vậy?",
    "Bạn do ai tạo ra?", "Bạn có thể giúp tôi những việc gì?", "Cảm ơn bạn nhiều nhé!",
    "Bạn giỏi thật đấy, cảm ơn nha", "Bạn tên là gì?", "Bạn hỗ trợ những lĩnh vực pháp luật nào?",
    "Bạn hoạt động dựa trên công nghệ gì vậy?", "Bạn có phải là luật sư không?",
    "ok bạn, tạm biệt nhé", "Giới thiệu về bản thân bạn đi", "Bạn làm được gì cho tôi?",
    "Chào bạn", "Cảm ơn, câu trả lời rất hữu ích", "Bạn thông minh ghê",
    "Bạn có thể giải đáp tình huống pháp lý thực tế không?", "Bạn khỏe không?",
    "Bạn giúp được gì cho tôi nhỉ?", "cảm ơn nhé", "Bạn có thể nói tiếng Việt không?",
    "Rất vui được trò chuyện với bạn",
]

REALTIME = [
    "Thời tiết hôm nay thế nào?", "Tỷ giá USD hôm nay bao nhiêu?",
    "Kết quả trận U23 Việt Nam tối qua thế nào?", "Hôm nay có tin tức gì mới về kinh tế không?",
    "Giá vàng SJC hôm nay bao nhiêu?", "Giá xăng hôm nay tăng hay giảm?",
    "Chứng khoán VN-Index hôm nay thế nào?", "Lịch thi đấu bóng đá tối nay có gì?",
    "Tin tức mới nhất về dịch bệnh hôm nay?", "Giá Bitcoin hiện tại bao nhiêu?",
    "Nhiệt độ ngoài trời bây giờ là bao nhiêu?", "Giá cà phê hôm nay thế nào?",
    "Tình hình giao thông Hà Nội giờ này ra sao?", "Phim nào đang hot ngoài rạp tuần này?",
    "Tin mới nhất về thời sự trong nước hôm nay?", "Dự báo bão sắp tới thế nào?",
    "Giá điện mới nhất hiện nay là bao nhiêu?", "Tin tức covid hôm nay thế nào?",
    "Đội tuyển Việt Nam đá với ai tối nay?", "Giá dầu thế giới hôm nay ra sao?",
]
REALTIME_RW = {
    "Thời tiết hôm nay thế nào?": "Thời tiết hôm nay như thế nào?",
}

# realtime web ON đủ thông tin (đã có địa điểm/đối tượng cụ thể)
REALTIME_WEB = [
    ("Thời tiết ở TP Hồ Chí Minh hôm nay thế nào?", "Thời tiết hôm nay tại Thành phố Hồ Chí Minh như thế nào?", ""),
    ("Giá vàng SJC hôm nay bao nhiêu một lượng?", "Giá vàng SJC hôm nay là bao nhiêu một lượng?", ""),
    ("Tin tức mới nhất về giá xăng dầu trong nước hôm nay?", "Tin tức mới nhất về giá xăng dầu trong nước hôm nay là gì?", ""),
    ("Tỷ giá USD sang VND hôm nay tại Vietcombank là bao nhiêu?", "Tỷ giá USD sang VND hôm nay tại ngân hàng Vietcombank là bao nhiêu?", ""),
    ("Kết quả trận Việt Nam gặp Thái Lan tối qua thế nào?", "Kết quả trận đấu giữa Việt Nam và Thái Lan tối qua là gì?", ""),
    ("Thời tiết Đà Nẵng cuối tuần này ra sao?", "Thời tiết tại Đà Nẵng vào cuối tuần này như thế nào?", ""),
    ("VN-Index đóng cửa hôm nay ở mức nào?", "Chỉ số VN-Index đóng cửa hôm nay ở mức bao nhiêu?", ""),
    ("Dự báo thời tiết Hà Nội 3 ngày tới thế nào?", "Dự báo thời tiết tại Hà Nội trong 3 ngày tới như thế nào?", ""),
    ("Giá xăng RON 95 hiện tại là bao nhiêu một lít?", "Giá xăng RON 95 hiện tại là bao nhiêu một lít?", ""),
    ("Lịch thi đấu vòng loại World Cup của Việt Nam sắp tới thế nào?", "Lịch thi đấu vòng loại World Cup sắp tới của đội tuyển Việt Nam như thế nào?", ""),
    ("Hà Nội", "Thời tiết hôm nay tại Hà Nội, Việt Nam như thế nào?",
     "## Các lượt gần nhất trong phiên:\nTrợ lý: Bạn đang ở thành phố/khu vực nào để mình tra thời tiết cho chính xác?"),
    ("Vàng nhẫn 9999 tại Hà Nội", "Giá vàng nhẫn 9999 hôm nay tại Hà Nội là bao nhiêu?",
     "## Các lượt gần nhất trong phiên:\nTrợ lý: Bạn muốn tra giá vàng loại nào và ở đâu?"),
]

# realtime web ON thiếu info → ask_user
REALTIME_ASK = [
    ("Thời tiết hôm nay thế nào?", "Thời tiết hôm nay như thế nào?", None),
    ("Mai có mưa không?", "Ngày mai trời có mưa không?", None),
    ("Giá vé máy bay đi đâu rẻ nhất bây giờ?", "Giá vé máy bay hiện tại đi đâu rẻ nhất?", ["ask_user", "web"]),
    ("Nhiệt độ bây giờ bao nhiêu độ?", "Nhiệt độ hiện tại là bao nhiêu độ?", None),
    ("Quán ăn nào ngon gần đây?", "Quán ăn nào ngon ở gần đây?", ["ask_user", "web"]),
    ("Trời có nắng không?", "Hôm nay trời có nắng không?", None),
    ("Tình hình giao thông giờ này thế nào?", "Tình hình giao thông hiện tại như thế nào?", None),
    ("Hôm nay có sự kiện gì gần đây không?", "Hôm nay có sự kiện gì diễn ra ở gần đây không?", ["ask_user", "web"]),
    ("Cuối tuần này thời tiết ra sao?", "Thời tiết cuối tuần này như thế nào?", None),
    ("Khách sạn nào tốt ở đó?", "Khách sạn nào tốt ở khu vực đó?", ["ask_user", "web"]),
    ("Bây giờ mấy giờ rồi?", "Bây giờ là mấy giờ?", ["ask_user", "web"]),
    ("Có chỗ nào vui chơi cuối tuần không?", "Có địa điểm vui chơi nào cho cuối tuần không?", ["ask_user", "web"]),
]

OOS = [
    "1 kg ổi bao nhiêu tiền?", "Công thức nấu phở bò ngon nhất là gì?",
    "Trường đại học sư phạm kỹ thuật có mấy khoa?", "Cách trồng cây xương rồng trong nhà thế nào?",
    "Làm sao để giảm cân nhanh trong 1 tuần?", "2 cộng 2 bằng mấy?",
    "Cách làm bánh kem tại nhà thế nào?", "Cách sửa máy tính bị treo thế nào?",
    "Tại sao bầu trời màu xanh?", "Thủ đô của nước Pháp là thành phố nào?",
    "Học lập trình Python bắt đầu từ đâu?", "Cách chăm sóc da mặt mùa đông thế nào?",
    "Sông dài nhất thế giới là sông nào?", "Làm sao để ngủ ngon hơn?",
    "Những địa điểm du lịch nổi tiếng ở Đà Lạt?", "Ai phát minh ra bóng đèn?",
    "Mẹo tiết kiệm điện trong gia đình là gì?", "Nên ăn gì để tăng cường sức đề kháng?",
    "Cách trồng rau sạch tại nhà thế nào?", "Cách luyện trí nhớ tốt hơn?",
    "Cách pha cà phê espresso ngon?", "Ai là người giàu nhất thế giới?",
    "Con mèo nhà tôi bị ốm phải làm sao?", "Học tiếng Anh hiệu quả thế nào?",
]
OOS_TUC = ["m câm đi m ngu quá", "đồ ngu", "câm mồm"]


def simple_rw(q):
    """Rewrite nhẹ cho câu non-legal đơn giản (chuẩn hoá dấu hỏi)."""
    q = q.strip()
    return q


def build_nonlegal(n_chitchat, n_realtime_reject, n_realtime_ask, n_realtime_web,
                   n_oos_reject, n_oos_web, rng):
    # chitchat (web off + on)
    pool = CHITCHAT[:]
    for i in range(n_chitchat):
        q = pool[i % len(pool)]
        web = (i % 4 == 0)
        add("chitchat", "chitchat", q, web, REALTIME_RW.get(q, simple_rw(q)),
            note="Chào hỏi/hỏi về bot.")

    # realtime reject (web OFF)
    for i in range(n_realtime_reject):
        q = REALTIME[i % len(REALTIME)]
        add("realtime_reject", "realtime", q, False, REALTIME_RW.get(q, simple_rw(q)),
            note="Realtime + web OFF → reject.")

    # realtime ask_user (web ON, thiếu info)
    for i in range(n_realtime_ask):
        q, rw, acc = REALTIME_ASK[i % len(REALTIME_ASK)]
        add("realtime_ask", "realtime", q, True, rw, needs_clar=True, acc_tools=acc,
            note="Realtime + web ON, thiếu info → ask_user.")

    # realtime web (web ON, đủ info)
    for i in range(n_realtime_web):
        q, rw, ctx = REALTIME_WEB[i % len(REALTIME_WEB)]
        add("realtime_web", "realtime", q, True, rw, ctx=ctx,
            note="Realtime + web ON, đủ info → web.")

    # oos reject (web OFF)
    allq = OOS + OOS_TUC
    for i in range(n_oos_reject):
        q = allq[i % len(allq)]
        add("oos_reject", "out_of_scope", q, False, simple_rw(q),
            note="Ngoài phạm vi + web OFF → reject.")

    # oos web (web ON)
    for i in range(n_oos_web):
        q = OOS[i % len(OOS)]
        add("oos_web", "out_of_scope", q, True, simple_rw(q),
            note="Ngoài phạm vi + web ON → web.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    args = ap.parse_args()
    rng = random.Random(SEED)

    corpus = load_corpus()
    print(f"Corpus sạch: {len(corpus)} (doc, article) pairs")

    # Phân bổ: legal ~72%, non-legal ~28% (legal là use-case chính)
    T = args.target
    n_legal = int(T * 0.72)
    n_ent = n_legal // 2
    n_noent = n_legal - n_ent

    n_rest = T - n_legal
    # chia phần non-legal
    n_chitchat = int(n_rest * 0.30)
    n_rt_reject = int(n_rest * 0.20)
    n_rt_ask = int(n_rest * 0.12)
    n_rt_web = int(n_rest * 0.13)
    n_oos_reject = int(n_rest * 0.13)
    n_oos_web = n_rest - (n_chitchat + n_rt_reject + n_rt_ask + n_rt_web + n_oos_reject)

    build_legal(corpus, n_ent, n_noent, rng)
    build_nonlegal(n_chitchat, n_rt_reject, n_rt_ask, n_rt_web, n_oos_reject, n_oos_web, rng)

    rng.shuffle(records)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Tổng số mẫu: {len(records)}")
    print("Theo intent     :", dict(Counter(r["expected_intent"] for r in records)))
    print("Theo tool       :", dict(Counter(r["expected_tool"] for r in records)))
    print("web ON / OFF    :", dict(Counter(r["web_search_enabled"] for r in records)))
    print("Legal theo domain:", dict(Counter(
        r["domain"] for r in records if r["expected_intent"] == "legal")))


if __name__ == "__main__":
    main()
