from abc import ABC, abstractmethod
import numpy as np

class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the provider"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the provider"""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return the vector of the embed list of strings"""
