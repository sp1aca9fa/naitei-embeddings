import numpy as np
from sentence_transformers import SentenceTransformer

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main() -> None:
    """Embeds 7 phrases to the model and prints a pair coisine similarity matrix"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [
        "I programmed my first software when I was 15.",
        "I went to a web development bootcamp a few months ago.",
        "I love french fries and pizza.",
        "He loves french fries and hamburgers.",
        "I played the new pizza delivery game for 2 hours, it's so much fun!",
        "Hard core gamer here hehe",
        "We play badminton sometimes when we have free time."
    ]

    embeddings = model.encode(sentences)

    print()
    print(f"norm of first embedding: {np.linalg.norm(embeddings[0])}")
    print(f"shape: {embeddings.shape}")
    print()

    print("       ", end="")
    for j in range(len(sentences)):
        print(f"  s{j}   ", end="")
    print()

    for i in range(len(embeddings)):
        print(f"  s{i}   ", end="")

        for j in range(len(embeddings)):
            print(f" {cosine(embeddings[i], embeddings[j]):.3f}", end=" ")
        print()


if __name__ == "__main__":
    main()
