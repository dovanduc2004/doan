import numpy as np
import torch
from pyvi import ViTokenizer
from source.model.extract_model import Bert_Extract
class Extract_Information():
    def __init__(self,bert:Bert_Extract):
        self.bert=bert
    def predict(self,contexts, question):
        lst_Answer_Final = {}  
        question=ViTokenizer.tokenize(question)
        try:
            for idx,(context) in enumerate(contexts):
                context=ViTokenizer.tokenize(context)
                inputs = self.bert.tokenizer(
                question,
                context,
                max_length=self.bert.max_length,
                truncation="only_second",
                stride=self.bert.stride,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = self.bert.model_extract(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
                
                start_logits = outputs.start_logits.squeeze().cpu().numpy()
                end_logits = outputs.end_logits.squeeze().cpu().numpy()
                offsets = inputs["offset_mapping"][0].cpu().numpy()
                
                answers = []
                start_indexes = np.argsort(start_logits)[-self.bert.nbert:][::-1].tolist()
                end_indexes = np.argsort(end_logits)[-self.bert.nbert:][::-1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if end_index < start_index or end_index - start_index + 1 > self.bert.max_answer_length:
                            continue
                        if offsets[start_index][0] is not None and offsets[end_index][1] is not None:
                            answer_text = context[offsets[start_index][0]: offsets[end_index][1]].strip()
                            if answer_text:
                                answer = {
                                    "text": answer_text,
                                    "score": start_logits[start_index] + end_logits[end_index],
                                }
                                answers.append(answer)
                
                if answers:
                    answers.sort(key=lambda x: x["score"], reverse=True)
                    best_answer = answers[0]['text']
                    lst_Answer_Final[idx] = best_answer.replace("_", " ").replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' :', ':').replace(' ;', ';')
                else:
                    lst_Answer_Final[idx] = "Không có câu trả lời"
        except Exception as e:
            print(f"Lỗi xảy ra: {e}")
            return "Không có câu trả lời do lỗi xử lý"
        return lst_Answer_Final