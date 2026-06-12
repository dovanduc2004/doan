from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()


def _default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Settings():
    def __init__(self):
        self.URL_QDRANT_LOCAL: str = os.getenv("URL_QDRANT_LOCAL")
        self.EXIST_COLLECTION_NAME: str = os.getenv("EXIST_COLLECTION_NAME")
        # LLM provider: "bedrock" (default) or "gemini"
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "bedrock").lower()
        # AWS Bedrock
        self.AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
        self.BEDROCK_MODEL_ID: str = os.getenv(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        # Gemini (kept for fallback / legacy)
        gemini_keys = os.getenv("APIS_GEMINI_LIST", "")
        self.APIS_GEMINI_LIST: list = [k for k in gemini_keys.split(',') if k]
        self.MODEL_GEMINI: str = os.getenv("MODEL_GEMIMI")
        self.MODEL_RERANK: str = os.getenv("MODEL_RERANKER")
        rerank_keys = os.getenv('API_RERANKER', "")
        self.API_RERANKER: list = [k for k in rerank_keys.split(',') if k]
        self.MODEL_EMBEDDING: str = os.getenv("MODEL_EMBEDDING")
        self.metadata_payload_key = "metadata"
        self.MODEL_EXTRACT: str = os.getenv("GENERATE_MODEL_EXTRACT")
        self.TOKENIZER: str = os.getenv("GENERATE_MODEL_TOKENIZER")
        self.MAX_LENGTH = 512
        self.STRIDE = 150
        self.N_BEST = 380
        self.MAX_ANSWER_LENGTH = 2000
        self.DEVICE = {"device": os.getenv("TORCH_DEVICE", _default_device())}
        self.YAML_PATH = os.getenv("PATH_PROMPT")
        self.PATH_INFOR = os.getenv("PATH_INFOR")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.DB_PATH = os.getenv(
            "DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "data", "chat_history.db"),
        )
        self.RERANK = os.getenv("MODEL_RERANK")
        # Agentic RAG tuning
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))
        self.GRADER_THRESHOLD = float(os.getenv("GRADER_THRESHOLD", "0.6"))
        self.HALLUCINATION_CHECK_ENABLED = os.getenv("HALLUCINATION_CHECK_ENABLED", "true").lower() == "true"
