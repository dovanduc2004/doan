from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from peft import PeftModel, PeftConfig
from source.core.config import Settings
class Bert_Extract():
    def __init__(self,config:Settings):
        self.max_length=config.MAX_LENGTH
        self.max_answer_length=config.MAX_ANSWER_LENGTH
        self.stride=config.STRIDE
        # self.config = PeftConfig.from_pretrained(config.MODEL_EXTRACT)
        # self.base_model = AutoModelForQuestionAnswering.from_pretrained(
        # self.config.base_model_name_or_path,torch_dtype=torch.float32) 
        # self.model_extract = PeftModel.from_pretrained(self.base_model,config.MODEL_EXTRACT)   
        self.model_extract = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_EXTRACT)
        self.tokenizer=AutoTokenizer.from_pretrained(config.TOKENIZER)
        self.nbert=config.N_BEST