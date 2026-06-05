from .base import EmbeddingProvider
import hashlib
import numpy as np

def _seed_from_text(text: str) -> int:
    """Hash text to a 32-bit unsigned int suitable for seeding numpy."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

class MockProvider(EmbeddingProvider):
    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            rng = np.random.default_rng(_seed_from_text(t))
            vec = rng.standard_normal(self.dimension)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors)
