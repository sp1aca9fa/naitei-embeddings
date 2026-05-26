import numpy as np
from sentence_transformers import SentenceTransformer

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the cosine similarity between two 1D vectors.
    Using only dot calculation, as long as the vectors are normalized to 1.0
    """
    return float(np.dot(a, b))

def print_cosine(sentences: list, name: str, embeddings: np.ndarray) -> None:
    """
    Print the cosine of each sentence against the entire setences list in matrix format with headers to help visualization.
    """
    print()
    print(f"Model {name}")
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

def main() -> None:
    """
    Using print_cosine, prints the cosine of a sentence against the entire sentences list as calculates by 2 different models.
    """
    model1 = SentenceTransformer("all-MiniLM-L6-v2")
    model2 = SentenceTransformer("intfloat/multilingual-e5-base")

    # sentences = [
    #     "Senior frontend engineer",
    #     "シニアフロントエンドエンジニア",
    #     "React　エンジニア",
    #     "Pokemon Cards",
    #     "ポケモンカード",
    #     "Trump is the new president of China",
    #     "トランプが中国の大統領に",
    # ]

    sentences = [
      "I have five years of experience building React frontends for fintech products.",
      "金融系プロダクトのReactフロントエンドを5年間構築してきた経験があります。",
      "I collect rare Pokemon trading cards from the original 1999 base set.",
      "1999年の初版のレアなポケモンカードを集めています。",
      "The new high-speed rail line between Tokyo and Sapporo will open in 2030.",
      "東京と札幌を結ぶ新しい新幹線路線は2030年に開業予定です。",
    ]

    embeddings_all_mini = model1.encode(sentences)
    embeddings_intfloat = model2.encode(sentences)

    # embeddings_intfloat = model2.encode(list(map(lambda s: "query: " + s, sentences)))

    embeddings_intfloat = model2.encode(
        [f"query: {s}" for s in sentences]
    )

    print_cosine(sentences, "all-MiniLM-L6-v2", embeddings_all_mini)
    print_cosine(sentences, "intfloat/multilingual-e5-base", embeddings_intfloat)

if __name__ == "__main__":
    main()
