from base import EmbeddingProvider
import hashlib
import numpy as np

def _seed_from_text(text: str) -> int:
    """Hash text to a 32-bit unsigned int suitable for seeding numpy."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

class MockProvider(EmbeddingProvider):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def name(self):
        return "mock"

    def dimension(self):
        return self.dimension

    def embed(self, texts):
        rng = np.random.default_rng(_seed_from_text(texts))
        vec = rng.standard_normal(dimension)
        vec /= np.linalg.norm(vec)
        return vec
