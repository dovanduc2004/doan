from source.core.config import Settings
from source.model.reset_apikey import APIKeyManager


class Gemini():
    """Holds config for the LLM generator.

    Name kept as 'Gemini' for import-compatibility with the rest of the
    codebase, but the active backend is AWS Bedrock when
    Settings.LLM_PROVIDER == 'bedrock'.
    """

    def __init__(self, setting: Settings):
        self.setting = setting
        self.provider = setting.LLM_PROVIDER
        self.model_gemini = setting.MODEL_GEMINI
        self.bedrock_model_id = setting.BEDROCK_MODEL_ID
        self.aws_region = setting.AWS_REGION
        if setting.APIS_GEMINI_LIST:
            self.key_manager = APIKeyManager(setting.APIS_GEMINI_LIST)
        else:
            self.key_manager = None
