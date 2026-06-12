from typing import List, Tuple
from source.model.rerank_model_finetune import RerankModelFinetune


class Rerank_Utils():
    """Reciprocal-rank-fusion + CrossEncoder rerank.

    Always uses the local fine-tuned CrossEncoder.
    """

    def __init__(self, model_finetune: RerankModelFinetune):
        self.model_finetune = model_finetune.model

    def reciprocal_rank_fusion(self, documents_nested, k=60):
        document_scores = {}
        try:
            for query_idx, result_list in enumerate(documents_nested):
                for rank, (doc, _) in enumerate(result_list):
                    doc_key = doc.page_content
                    doc_metadata = doc.metadata
                    rrf_score = 1 / (k + rank + 1)
                    if doc_key not in document_scores:
                        document_scores[doc_key] = {
                            "score": 0,
                            "doc_metadata": doc_metadata,
                            "count": 0,
                        }
                    document_scores[doc_key]["score"] += rrf_score
                    document_scores[doc_key]["count"] += 1
            reranked_docs = sorted(document_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            return reranked_docs[:50]
        except Exception as e:
            print(f"Lỗi khi tính RRF: {e}")
            return []

    def rerank_documents_finetune(self, query, documents) -> List[Tuple[str, dict]]:
        if len(documents) > 50:
            raise ValueError("Số lượng documents không được vượt quá 50.")

        doc_contents = [doc for doc, _ in documents]
        doc_metadata = [info['doc_metadata'] for _, info in documents]
        doc_contents = [
            d.replace("_", " ")
             .replace(' .', '.').replace(' ,', ',')
             .replace(' !', '!').replace(' ?', '?')
             .replace(' :', ':').replace(' ;', ';')
            for d in doc_contents
        ]
        try:
            pairs = [(query, doc) for doc in doc_contents]
            scores = self.model_finetune.predict(pairs)
            scores = [float(score) for score in scores]
            ranked = sorted(zip(scores, range(len(doc_contents))), key=lambda x: x[0], reverse=True)
            return [
                (doc_contents[idx], {'score': score, 'doc_metadata': doc_metadata[idx]})
                for score, idx in ranked[:5]
            ]
        except Exception as e:
            print(f"An error occurred: {e}")
            return []