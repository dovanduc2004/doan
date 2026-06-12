from source.core.config import Settings
from source.model.embedding_model import Sentences_Transformer_Embedding
from langchain_qdrant import Qdrant
class Qdrant_Vector():
    def __init__(self,setting:Settings,embedding:Sentences_Transformer_Embedding):
        self.setting=setting
        self.embedding=embedding
    def Open_Qdrant(self):
        # check_compatibility=False: server Qdrant hiện tại (1.17.x) có
        # thể mới hơn client (1.14.x) — langchain-qdrant warn mỗi lần mở
        # connection, tắt check để giảm spam log. Nếu sau này có breaking
        # API change, upgrade qdrant-client trong requirements.txt.
        open_collection = Qdrant.from_existing_collection(
            embedding = self.embedding.embeddings_bkai,
            url =self.setting.URL_QDRANT_LOCAL,
            collection_name =self.setting.EXIST_COLLECTION_NAME,
            metadata_payload_key=self.setting.metadata_payload_key,
            check_compatibility=False,
        )
        return open_collection


