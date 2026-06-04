import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

provider_name = os.getenv("EMBEDDING_PROVIDER", "mock")
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_model = os.getenv("HUGGINGFACE_MODEL", "intfloat/multilingual-e5-base")
mock_dimension = os.getenv("MOCK_DIMENSION", 768)
