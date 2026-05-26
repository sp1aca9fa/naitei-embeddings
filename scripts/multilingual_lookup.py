import numpy as np
from sentence_transformers import SentenceTransformer


def top_k(query_vec: np.ndarray, candidate_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """
    Using doc calculation (query @ candidates), evaluates queries against candidates and returns top 3 candidates indeces and scores.
    """
    scores = candidate_vecs @ query_vec

    result = [(index, score) for index, score in enumerate(scores)]

    return sorted(result, key=lambda x: x[1], reverse=True)[:k]

def print_results(query: str, k: int) -> None:
    """
    Print assist to help organize printing on console.
    """
    print()
    print("Query:", query)
    print(f"Top {k} candidates:")

def main() -> None:
    """
    Using dot calculation/dot product (query @ candidates), evaluates queries against candidates and print top 3 candidates for each query.
    """
    queries = [
        "React frontend developer",
        "data engineer with Python and SQL",
        "DevOps engineer comfortable with Kubernetes"
    ]

    candidates = [
        "フロントエンドエンジニア募集 React TypeScript 経験者歓迎",
        "データエンジニア Python SQL BigQuery 経験必須",
        "インフラエンジニア AWS Kubernetes Docker 運用経験",
        "営業職 BtoB 法人営業 経験3年以上",
        "バックエンドエンジニア Go 実務経験3年以上",
        "モバイルアプリエンジニア iOS Swift 開発経験歓迎",
        "AIエンジニア 機械学習 Python PyTorch 実務経験",
        "プロジェクトマネージャー IT業界 PM経験 アジャイル開発経験"
    ]

    model = SentenceTransformer("intfloat/multilingual-e5-base")

    emb_queries = model.encode([f"query: {query}" for query in queries])
    emb_candidates = model.encode([f"query: {candidate}" for candidate in candidates])

    print()
    print(f"queries shape: {emb_queries.shape}")
    print(f"candidates shape: {emb_candidates.shape}")
    print(f"emb_queries[0] norm: {np.linalg.norm(emb_queries[0])}")
    print(f"emb_candidates[0] norm: {np.linalg.norm(emb_candidates[0])}")

    for q in range(len(queries)):
        print_results(queries[q], 3)
        for index, score in top_k(emb_queries[q], emb_candidates, 3):
            print(score, candidates[index])


if __name__ == "__main__":
    main()
