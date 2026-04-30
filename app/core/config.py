from pydantic_settings import BaseSettings
from decouple import config

class Settings(BaseSettings):

    API_BASE_URL                    : str = config('API_BASE_URL', cast=str)
    ENV                             : str = config('ENV', default='dev', cast=str)

    TELEGRAM_BOT_TOKEN              : str = config('TELEGRAM_BOT_TOKEN', cast=str)
    TELEGRAM_ALLOWED_CHAT_IDS       : list[int] = config('TELEGRAM_ALLOWED_CHAT_IDS', default='', cast=lambda v: [int(i.strip()) for i in v.split(',') if i.strip()])
    TELEGRAM_ALLOWED_THREAD_IDS     : list[int] = config('TELEGRAM_ALLOWED_THREAD_IDS', default='', cast=lambda v: [int(i.strip()) for i in v.split(',') if i.strip()])

    # Ollama (active agent)
    OLLAMA_BASE_URL                 : str = config('OLLAMA_BASE_URL', default='http://localhost:11434', cast=str)
    OLLAMA_MODEL                    : str = config('OLLAMA_MODEL', default='qwen3.5:0.8b', cast=str)

    # HuggingFace (kept, not active)
    HF_MODEL_ID                     : str = config('HF_MODEL_ID', default='google/gemma-4-e2b-it', cast=str)
    HF_TOKEN                        : str = config('HF_TOKEN', default='', cast=str)

    OCR_MCP_URL                     : str = config('OCR_MCP_URL', default='http://localhost:8001/mcp', cast=str)
    HTTP_MCP_URL                    : str = config('HTTP_MCP_URL', default='http://localhost:8002/mcp', cast=str)
    ALPR_MCP_URL                    : str = config('ALPR_MCP_URL', default='http://localhost:8003/mcp', cast=str)


settings = Settings()