from base import EmbeddingProvider
import numpy as np
from sentence_transformers import SentenceTransformer

class HuggingFaceProvider(EmbeddingProvider):
    def __init__(self, model_name = "intfloat/multilingual-e5-base": str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = model.get_sentence_embedding_dimension()

    def name(self):
        return self.model_name

    def dimension(self):
        return self.dimension

    def embed(self, texts):
        embeddings = self.model.encode([f"query: {t}" for t in texts], prefixed, normalize_embeddings=True)
        return embeddings
