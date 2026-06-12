import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login, snapshot_download




HF_TOKEN = "Your_HF_TOKEN"


# --- Rerank model ---
HF_REPO_ID = "bhuy71/rerank_model"
MODEL_DIR  = "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main/rerank_model"
TOKENIZER_BASE = None     

# --- BERT QA extractor ---
# HF_REPO_ID = "bhuy71/bert_extract_full_fine-tuned"
# MODEL_DIR  = "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main/bert_question_answer"
# TOKENIZER_BASE = "google-bert/bert-base-multilingual-cased"

# --- BERT QA extractor ---

# HF_REPO_ID = "bhuy71/bert_extract_full_fine-tuned"
# MODEL_DIR  = "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main/model_extractor"
# TOKENIZER_BASE = "google-bert/bert-base-multilingual-cased"

# --- Embedding model ---
# HF_REPO_ID = "bhuy71/embedding_finetuned"
# MODEL_DIR  = "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main/checkpoint-276"
# TOKENIZER_BASE = None

PRIVATE = False

# các file/folderkhông cần push lên HF
IGNORE_PATTERNS = [
]

# File tokenizer cần fetch từ base model khi TOKENIZER_BASE được set
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "special_tokens_map.json",
    "added_tokens.json",
]

# ============================================


def bundle_tokenizer_from_base(model_dir: Path, base_repo: str):
    """Download tokenizer files từ base model repo và copy vào model_dir.

    Chỉ cần khi folder model chỉ có weight + config (không có tokenizer files).
    """
    existing = {p.name for p in model_dir.iterdir() if p.is_file()}
    # Nếu đã có tokenizer.json thì skip
    if "tokenizer.json" in existing or "tokenizer_config.json" in existing:
        print(f"✓ Tokenizer đã có sẵn trong {model_dir.name}/, bỏ qua fetch.")
        return

    print(f"→ Folder không có tokenizer, fetch từ {base_repo}...")
    snapshot_path = Path(snapshot_download(
        repo_id=base_repo,
        allow_patterns=TOKENIZER_FILES,
        token=HF_TOKEN,
    ))
    copied = []
    for p in snapshot_path.iterdir():
        if p.is_file() and p.name in TOKENIZER_FILES:
            dest = model_dir / p.name
            dest.write_bytes(p.read_bytes())
            copied.append(p.name)
    print(f"✓ Copied {len(copied)} file tokenizer: {copied}")


def is_ignored(rel_path: Path) -> bool:
    s = str(rel_path)
    return any(
        fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(rel_path.parts[0], pat.rstrip("/*"))
        for pat in IGNORE_PATTERNS
    )


def main():
    model_path = Path(MODEL_DIR).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"ERROR: MODEL_DIR không tồn tại: {model_path}")

    # Bundle tokenizer nếu cần
    if TOKENIZER_BASE:
        bundle_tokenizer_from_base(model_path, TOKENIZER_BASE)

    all_files = sorted(
        p.relative_to(model_path) for p in model_path.rglob("*") if p.is_file()
    )
    kept = [f for f in all_files if not is_ignored(f)]
    skipped = [f for f in all_files if is_ignored(f)]

    total_mb = sum((model_path / f).stat().st_size for f in kept) / 1024 / 1024
    print(f"\n📂 Folder:  {model_path}")
    print(f"📦 {len(kept)} file sẽ push, tổng {total_mb:.1f} MB:")
    for fn in kept:
        size_mb = (model_path / fn).stat().st_size / 1024 / 1024
        print(f"   ✓ {str(fn):<50} {size_mb:>8.2f} MB")
    if skipped:
        print(f"\n🚫 {len(skipped)} file BỎ QUA:")
        for fn in skipped[:10]:
            print(f"   - {fn}")
        if len(skipped) > 10:
            print(f"   ... và {len(skipped) - 10} file khác")
    print()

    kept_names = {str(f) for f in kept}
    has_weight = any(n in kept_names for n in ("model.safetensors", "pytorch_model.bin"))
    has_config = "config.json" in kept_names
    has_tokenizer = any(n in kept_names for n in TOKENIZER_FILES)
    if not has_weight:
        print("⚠  Warning: không thấy file weight (model.safetensors / pytorch_model.bin)")
    if not has_config:
        print("⚠  Warning: không thấy config.json")
    if not has_tokenizer:
        print("⚠  Warning: không thấy bất kỳ file tokenizer nào — runtime sẽ fail load")

    confirm = input(f"Push lên  https://huggingface.co/{HF_REPO_ID}  ? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Hủy.")
        return

    login(token=HF_TOKEN)
    create_repo(HF_REPO_ID, private=PRIVATE, exist_ok=True, token=HF_TOKEN)
    print(f"✓ Repo sẵn sàng")

    api = HfApi()
    commit_info = api.upload_folder(
        folder_path=str(model_path),
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Upload {model_path.name} from local",
    )
    print(f"\n✓ Push xong!")
    print(f"   Commit:  {commit_info.commit_url}")
    print(f"   Repo:    https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
