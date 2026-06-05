import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.providers import get_provider
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the cosine similarity between two 1D vectors.
    Using only dot calculation, as long as the vectors are normalized to 1.0
    """
    return float(np.dot(a, b))

provider = get_provider()
print(f"Provider name: {provider.name}")
print(f"Provider dimension: {provider.dimension}")

sentences = ["React developer", "Reactエンジニア", "I made pasta"]
embedding_test = provider.embed(sentences)

print(f"Embedding shape: {embedding_test.shape}")
print(f"Norm of the first embedding: {np.linalg.norm(embedding_test[0])}")

print("       ", end="")
for j in range(len(sentences)):
    print(f"  s{j}   ", end="")
print()

for i in range(len(embedding_test)):
    print(f"  s{i}   ", end="")

    for j in range(len(embedding_test)):
        print(f" {cosine(embedding_test[i], embedding_test[j]):.3f}", end=" ")
    print()
