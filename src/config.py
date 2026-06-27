import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

embedding_provider = os.getenv("EMBEDDING_PROVIDER", "mock")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
openai_dimension = int(os.getenv("OPENAI_DIMENSION", 1536))
huggingface_model = os.getenv("HUGGINGFACE_MODEL", "intfloat/multilingual-e5-base")
mock_dimension = int(os.getenv("MOCK_DIMENSION", 768))
database_url = os.getenv("DATABASE_URL")
