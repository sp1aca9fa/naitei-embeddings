from .base import EmbeddingProvider
from sentence_transformers import SentenceTransformer
import numpy as np

class HuggingFaceProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode([f"query: {t}" for t in texts], normalize_embeddings=True)
        return embeddings
