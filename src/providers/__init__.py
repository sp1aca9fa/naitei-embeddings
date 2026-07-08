from .base import EmbeddingProvider
from .mock import MockProvider
from .huggingface import HuggingFaceProvider
from .openai import OpenAIProvider
from .. import config

def get_provider(provider_name: str | None = None) -> EmbeddingProvider:
    name =  provider_name or config.embedding_provider

    if name == "mock":
        return MockProvider(config.mock_dimension)
    if name == "huggingface":
        return HuggingFaceProvider()
    if name == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OpenAI API key not set up")
        return OpenAIProvider(api_key = config.openai_api_key, model_name = config.openai_model, dimension = config.openai_dimension)
    raise ValueError(f"Unknown provider")
