from openai import OpenAI
from .base import EmbeddingProvider
import numpy as np

class OpenAIProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", dimension:int = 1536):
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "openai/" + self.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return np.array([d.embedding for d in response.data])
