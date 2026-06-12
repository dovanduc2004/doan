# Eval System 

## Dùng cho HỆ THỐNG Agentic RAG

| File | Vai trò |
|---|---|
| `build_eval_dataset.py` | Gộp 5 file QA theo domain (`data/data_processed/*.csv`) thành `eval_system_dataset.csv` (cột: domain, question, answer). |
| `eval_system.py` | Chạy Orchestrator hiện tại (Planner→Retrieve→Grade→Generate→Check) trên từng câu hỏi, lấy event `final` làm câu trả lời, rồi chấm **cosine (embedding nội bộ) + ROUGE-L + token-F1** so với đáp án vàng. Có `run` / `score` / `all`, cache ra `eval_system_predictions.csv`, report `eval_system_report.json`. |
| `eval_system_dataset.csv` | Dataset eval (722 QA pháp lý, 5 domain khớp Qdrant). |

### Cách chạy
```bash
# (1) build dataset (1 lần)
python build_eval_dataset.py

# (2) chạy hệ thống + chấm điểm (cần AWS credentials cho Bedrock + Qdrant đang chạy)
python eval_system.py all

# thử nhanh 20 mẫu
python eval_system.py run --limit 20 && python eval_system.py score

# bật web fallback khi RAG thiếu dữ liệu
python eval_system.py run --web
```



