"""Gộp 5 file QA theo domain thành 1 dataset eval cho hệ thống hiện tại.

5 domain QA này KHỚP ĐÚNG 5 domain đã embed vào Qdrant:
  honnhan.csv            -> honNhan_GiaDinh
  atgt.csv               -> giaoThong_VanTai   (an toàn giao thông)
  giaoduc.csv            -> giao_duc
  laodong.csv            -> laoDong_TienLuong
  trachnhiem_hinhsu.csv  -> trachNhiem_HinhSu

Mỗi dòng: question (đã bỏ tiền tố "Câu N."), answer (đáp án vàng), domain.
Đây là datang dùng để eval nhánh RAG (legal) end-to-end.

Chạy:
    python build_eval_dataset.py
=> ghi eval_system_dataset.csv
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "data_processed"
OUT = HERE / "eval_system_dataset.csv"

DOMAIN_FILES = {
    "honNhan_GiaDinh": "honnhan.csv",
    "giaoThong_VanTai": "atgt.csv",
    "giao_duc": "giaoduc.csv",
    "laoDong_TienLuong": "laodong.csv",
    "trachNhiem_HinhSu": "trachnhiem_hinhsu.csv",
}

# bỏ tiền tố "Câu 12." / "Câu 12:" / "12." ở đầu câu hỏi
_PREFIX = re.compile(r"^\s*(Câu\s*\d+\s*[\.:\)]?\s*)", re.IGNORECASE)


def clean_question(q: str) -> str:
    q = str(q).strip()
    q = _PREFIX.sub("", q)
    return q.strip()


def main():
    frames = []
    for domain, fname in DOMAIN_FILES.items():
        fp = PROCESSED / fname
        if not fp.exists():
            print(f"[!] thiếu {fname}, bỏ qua")
            continue
        df = pd.read_csv(fp)
        df = df[["question", "answer"]].copy()
        df["question"] = df["question"].apply(clean_question)
        df["answer"] = df["answer"].astype(str).str.strip()
        df = df[(df["question"].str.len() > 0) & (df["answer"].str.len() > 0)]
        df = df.drop_duplicates(subset=["question"])
        df["domain"] = domain
        frames.append(df)
        print(f"{domain:<20} {len(df)} mẫu")

    out = pd.concat(frames, ignore_index=True)
    out = out[["domain", "question", "answer"]]
    out.to_csv(OUT, index=False)
    print(f"\nTổng: {len(out)} mẫu -> {OUT.name}")


if __name__ == "__main__":
    main()
