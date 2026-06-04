from openai import OpenAI
from base import EmbeddingProvider
import numpy as np
from sentence_transformers import SentenceTransformer

class OpenAIProvider(EmbeddingProvider):
    def __init__(self, api_key: str):
        self.model_name = "openai/text-embedding-3-small"
        self.model = SentenceTransformer(model_name)
        self.dimension = model.get_sentence_embedding_dimension()

    def name(self):
        return self.model_name

    def dimension(self):
        return self.dimension

    def embed(self, texts):
        embeddings = self._client.embeddings.create(model="text-embedding-3-small", input=texts)
        return embeddings
