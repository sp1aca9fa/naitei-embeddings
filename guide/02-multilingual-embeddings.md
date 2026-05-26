# Chapter 2: Multilingual Embeddings

## What we're building in this chapter

Two more exploration scripts under `scripts/`. The first swaps the English-only MiniLM model for `intfloat/multilingual-e5-base` and shows what changes when you feed it Japanese alongside English. The second builds a tiny cross-language nearest-neighbor lookup: given a query in one language, find the closest entries from a list written in another. Neither script becomes part of the FastAPI service. They exist so that by the end of the chapter you have personally watched a multilingual model do the one thing this entire project depends on, which is recognize that "React developer" and "Reactエンジニア" mean the same thing.

## Why this matters

The naitei-embeddings service exists because Naitei has English CVs and Japanese job descriptions and needs to compare them. If embeddings only worked within one language, the project would be pointless. The whole bet is that a multilingual model can place "5 years of TypeScript experience" and "TypeScript5年以上の経験" near each other in a shared vector space, with no translation step in between.

This chapter is where that bet stops being a claim from a tutorial and becomes something you have seen with your own eyes. You will also see the negative result: what an English-only model like MiniLM does when you give it Japanese. The two side-by-side similarity matrices make the case better than any explanation.

There is also a practical concern. The model you adopt here (`intfloat/multilingual-e5-base`) has a quirk that catches a lot of people: it expects you to prepend `query:` or `passage:` to your input strings. Skipping this gives you noticeably worse similarity scores and there is no error or warning. It is exactly the kind of thing that erodes a few percent of accuracy across an entire production system if you do not learn it early.

## Concepts

### Why most embedding models are English-only

Embedding models are neural networks, and like all neural networks they are limited by what they were trained on. The training data for `all-MiniLM-L6-v2` was a few billion English sentence pairs. The model never saw Japanese during training, so it has no learned representation for Japanese semantics.

The story does not stop there. Before any text reaches the network, it is broken into tokens by a tokenizer. The tokenizer has its own fixed vocabulary, also built from the training data. MiniLM's vocabulary is built around English words, English subword fragments, and a small set of punctuation. When the tokenizer encounters a Japanese character it has never seen, it falls back to byte-level fragments or the special `[UNK]` "unknown" token. The model then sees a sequence of nearly content-free tokens.

The result is that all Japanese inputs end up encoded in roughly the same way: as long runs of low-information fallback tokens. The model's similarity output for two unrelated Japanese sentences will often be surprisingly high, not because the model thinks they are about similar topics, but because it cannot tell them apart. They look the same to it, the way two pages of an unfamiliar script look the same to a person who cannot read it.

This is the first surprise of the chapter. When you run an English-only model on Japanese text, you do not get an error. You get smooth, plausible-looking numbers that are actually meaningless. Worse, they often look high, which is the opposite of what you would want if you were trying to detect that the model has no signal.

### What multilingual models do differently

A multilingual model is trained on text from many languages at once, with at least some of the training objective explicitly pushing translations of the same sentence toward each other in the output space. The data typically includes parallel corpora (the same sentence written in two languages) and the contrastive loss says "these two should be near each other; these two should be far apart".

The result is a single shared vector space across all the languages the model has seen. "React developer" in English, "Reactエンジニア" in Japanese, and "Desarrollador de React" in Spanish all land near each other, because the training process directly rewarded that arrangement. None of them lands near "I made pasta for dinner" or "夕食にパスタを作った", because the training penalized that.

The number of languages a multilingual model handles well varies. Some are bilingual by design (English plus Chinese, say). Others, including `multilingual-e5-base`, claim around 100 languages with varying quality. Japanese and English are both strongly represented because they have a lot of parallel corpora available.

### The model: `intfloat/multilingual-e5-base`

This is the model you will use for the rest of the project. The key facts to internalize:

- **Output dimension: 768.** Not 384 like MiniLM. You cannot mix vectors from the two models in the same comparison; they live in different vector spaces.
- **Size on disk: about 280 MB.** Roughly 3.5x larger than MiniLM. The first download will take a minute or two on a normal connection.
- **Strong on Japanese and English.** This is the reason it was chosen for the project. It also handles many other languages, which is irrelevant to us but does not hurt.
- **It is normalized.** Output vectors have L2 norm very close to 1.0, so cosine similarity is just the dot product. Same property MiniLM had.
- **It needs prefixes.** This is the quirk worth its own section.

### The `query:` / `passage:` prefix quirk

E5 models were trained on a retrieval task: given a question, find the passage that answers it. To help the model learn, the training data labeled each text with its role, either "query" (the question) or "passage" (a candidate answer). The way this labeling was done is the literal string prefix `query:` or `passage:` prepended to the text before it goes into the model.

The result is that E5 expects you to do the same at inference time. If you embed `"React developer"` directly, the model produces a vector. If you embed `"query: React developer"`, the model produces a slightly different and usually better-calibrated vector for retrieval-style tasks. There is no error if you skip the prefix. The numbers just come out worse, and you have no way to tell from looking at them.

The rules in practice:

- **Asymmetric retrieval** (you have a question and a set of documents): prefix the question with `query:` and each document with `passage:`. This is the case the model was trained for.
- **Symmetric similarity** (you are comparing two things of the same kind, like two CV bullets, or two job titles): prefix both with `query:`. The E5 authors recommend this and the leaderboard numbers back it up.
- **Always prefix.** There is no case where omitting the prefix is correct for E5.

For this project, the most common case is symmetric: matching CV bullets against JD sentences. Both get `query:`. The skill canonicalization endpoint (Chapter 7) is borderline asymmetric (input mention vs. canonical list), and we will revisit which prefix to use when we get there. For now, just commit `query:` to muscle memory.

If the prefix feels weird, remember it is not a special token, not a control character, not magic. It is the literal six characters `q`, `u`, `e`, `r`, `y`, `:`, followed by a space, prepended to your text. The model learned during training that text starting that way plays the role of a query. That is the entire mechanism.

### A note on similarity score ranges

In Chapter 1 the guide said cosine similarities "cluster between about 0.1 and 0.95". That was a half-truth. MiniLM, the model from Chapter 1, does occasionally produce small negative values (typically in the -0.2 to 0.0 range) for genuinely unrelated sentences. This is normal and not a sign of a bug.

E5 is different. Because of how it was trained (contrastive learning with explicit normalization and a tighter loss), its outputs almost always sit in the positive range, even for very unrelated text. You will rarely see negatives from E5. The practical cluster:

- 0.6 to 0.7: typical floor for "two unrelated sentences in the same language". Higher than MiniLM's floor.
- 0.7 to 0.8: same topic, different details.
- 0.8 to 0.9: near-paraphrase.
- 0.9+: very close paraphrase or identical.

This higher floor is not a problem, it is just a different calibration. When you tune similarity thresholds in later chapters, the cutoff for "this CV bullet covers this JD requirement" will be in the 0.75 to 0.85 range with E5, not 0.5 like it might be with MiniLM. The absolute numbers do not matter; the relative spacing between matched and unmatched pairs is what does the work.

## The tools we're using

### `sentence-transformers` (same library, new model)

You already have it installed from Chapter 1. No new dependency. The change is the model name string passed to `SentenceTransformer(...)`.

- `SentenceTransformer("intfloat/multilingual-e5-base")` downloads the e5 model on first run, caches it under `~/.cache/huggingface/`, and returns the same kind of object as before.
- `model.encode(["query: ..."])` works exactly like in Chapter 1. The prefix is just part of the input string.

### The model: `intfloat/multilingual-e5-base`

- What it is: a multilingual sentence embedding model from Microsoft Research, published 2022, fine-tuned for retrieval with the contrastive E5 method.
- What it does for us: produces a 768-dimensional vector for any text in any of the ~100 supported languages, where vectors for the same meaning across languages end up near each other.
- Size: ~280 MB on disk after download.
- Hugging Face page: <https://huggingface.co/intfloat/multilingual-e5-base>
- Paper (skim, do not feel obligated): <https://arxiv.org/abs/2402.05672>

### `numpy` (still)

Same role as Chapter 1. Holds the vectors, lets you take dot products and norms cheaply. Nothing new to learn this chapter.

## How it fits together

For Exercise 1 (replication), the flow is:

```
list of bilingual sentence pairs
        |
        +-----------------+
        |                 |
        v                 v
load MiniLM         load multilingual-e5
        |                 |
        v                 v
encode all sentences      encode all sentences
(no prefix)               (each prefixed with "query:")
        |                 |
        v                 v
N x N cosine matrix       N x N cosine matrix
        |                 |
        v                 v
print side by side, observe the difference
```

For Exercise 2 (application), the flow is:

```
list of EN "queries" (e.g., role titles)
list of JA "candidates" (e.g., job posting phrases)
        |
        v
prefix everything with "query:"
        |
        v
encode queries -> matrix shape (Nq, 768)
encode candidates -> matrix shape (Nc, 768)
        |
        v
for each query:
    compute cosine with every candidate
    sort candidates by score, descending
    print top 3
```

Neither flow touches FastAPI, `src/`, or the database. The point of Chapter 2 is to establish the multilingual primitive in isolation. Wiring it into the service starts in Chapter 3 (provider abstraction) and continues in Chapter 5 (the match endpoint).

## Code examples

### Example 1: Loading the multilingual model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")
vector = model.encode("query: hello world")

print(vector.shape)   # (768,)
print(vector.dtype)   # float32
```

First run downloads ~280 MB. Subsequent runs load from the local cache and take a second or two.

### Example 2: The prefix in action

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

# Same content, with and without the prefix
plain = model.encode("React developer")
prefixed = model.encode("query: React developer")

ja_plain = model.encode("Reactエンジニア")
ja_prefixed = model.encode("query: Reactエンジニア")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"no prefix:    {cosine(plain, ja_plain):.3f}")
print(f"with prefix:  {cosine(prefixed, ja_prefixed):.3f}")
```

Both pairs should score high (the meaning genuinely is the same), but the prefixed version typically scores a bit higher and is what the model was designed for. Run this once so you have a concrete number in mind for what "a bit higher" means.

### Example 3: Cross-language batch

```python
sentences = [
    "query: senior backend engineer",
    "query: シニアバックエンドエンジニア",
    "query: junior frontend developer",
    "query: ジュニアフロントエンド開発者",
    "query: I made pasta for dinner",
    "query: 夕食にパスタを作った",
]
embeddings = model.encode(sentences)
print(embeddings.shape)   # (6, 768)
```

When you compute the pairwise cosine matrix on these, you should see three clear pairs of high scores along the diagonal-offset positions (rows 0-1, 2-3, 4-5) and lower scores everywhere else. That is the multilingual property doing its job.

### Example 4: A minimal nearest-neighbor lookup

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

queries = ["query: React frontend role"]
candidates = [
    "query: フロントエンドエンジニア React 経験者募集",
    "query: バックエンド Python エンジニア",
    "query: データサイエンティスト Python R 経験",
    "query: モバイルアプリ開発 Swift Kotlin",
]

q_vec = model.encode(queries)[0]            # shape (768,)
c_vecs = model.encode(candidates)           # shape (4, 768)

# All vectors are unit-normalized, so dot product = cosine.
scores = c_vecs @ q_vec                      # shape (4,)

ranked = sorted(zip(scores, candidates), reverse=True)
for score, text in ranked:
    print(f"{float(score):.3f}  {text}")
```

Two things to notice: the dot product `@` operator does the cosine math in one matrix multiplication because the vectors are already normalized, and the top result is the Japanese frontend posting even though the query is in English. That single observation is what the rest of this project is built on.

## Your tasks

Both exercises produce throwaway scripts under `scripts/`. Keep them committed; they are your learning record.

### Exercise 1: Replication

Build a script that proves to you, visually, that the multilingual model handles Japanese and the English-only model does not.

1. **Create `scripts/learn_multilingual.py`.**

2. **Inside the script:**
   - Load both `all-MiniLM-L6-v2` and `intfloat/multilingual-e5-base`. The first download of the multilingual model will take a minute or two.
   - Define a list of at least six sentences, arranged as three bilingual pairs. Each pair should mean roughly the same thing in English and Japanese. Pick topics with some variety (a role title, a hobby, a piece of news). Example pairing pattern:
     - sentence 0: English version of topic A
     - sentence 1: Japanese version of topic A
     - sentence 2: English version of topic B
     - sentence 3: Japanese version of topic B
     - sentence 4: English version of topic C
     - sentence 5: Japanese version of topic C
   - Embed the list twice: once with MiniLM (no prefix, MiniLM does not use prefixes), once with `multilingual-e5-base` (with `query:` prefixed to every sentence).
   - Reuse your `cosine(a, b)` function from `learn_embeddings.py`. Copy it into this script rather than importing; the scripts are independent exploration tools.
   - Print two N by N similarity matrices, one for each model, labeled clearly so you can compare them side by side.

3. **Look at the two matrices and notice:**
   - In the e5 matrix, the cells `(0,1)`, `(2,3)`, `(4,5)` should be the highest in their rows (or close to it). That is the multilingual property.
   - In the MiniLM matrix, those same cells will look much weaker. Often the Japanese sentences will look more similar to *each other* than to their English counterparts, because MiniLM sees them all as the same gibberish.
   - The absolute floor of the e5 matrix sits higher than MiniLM's (often above 0.7 even for unrelated pairs). That is the calibration shift this chapter mentioned.

4. **Verify shapes and norms** with one print line each: confirm e5 embeddings have shape `(N, 768)` and that `np.linalg.norm(emb[0])` is close to 1.0.

### Exercise 2: Application

Build a small cross-language nearest-neighbor lookup. The goal: practice using embeddings to *answer a question* rather than just observe a matrix. The script is unrelated to the FastAPI service for now.

The scenario: you have a short list of English skill or role descriptions ("queries") and a longer list of Japanese job posting fragments ("candidates"). For each query, you want the top 3 most similar candidates with their scores.

1. **Create `scripts/multilingual_lookup.py`.**

2. **Inside the script:**
   - Load `intfloat/multilingual-e5-base` only. MiniLM is not needed here.
   - Define a `queries` list of 3 to 5 English strings. Suggested examples (use your own if you prefer):
     - `"React frontend developer"`
     - `"data engineer with Python and SQL"`
     - `"DevOps engineer comfortable with Kubernetes"`
   - Define a `candidates` list of 8 to 12 Japanese strings that look like the kind of phrasing you would find on a Japan tech job posting. Mix in a few that should match each query well, and a few that should not match anything. You do not need real job postings; invent plausible phrases. A few examples to seed your imagination:
     - `"フロントエンドエンジニア募集 React TypeScript 経験者歓迎"`
     - `"データエンジニア Python SQL BigQuery 経験必須"`
     - `"インフラエンジニア AWS Kubernetes Docker 運用経験"`
     - `"営業職 BtoB 法人営業 経験3年以上"`
   - Prefix every string in both lists with `query:` before embedding (symmetric similarity case).
   - Write a function `top_k(query_vec: np.ndarray, candidate_vecs: np.ndarray, k: int) -> list[tuple[int, float]]` that returns the indices and scores of the top-k candidates by cosine similarity, sorted descending. Use the fact that the vectors are unit-normalized to compute all scores in one matrix multiplication (`candidate_vecs @ query_vec`). Type hints and a docstring as usual.
   - For each query, print:
     - The query text.
     - The top 3 candidates as `score (3 decimal places)  candidate_text`.
     - A blank line between queries.

3. **Verify the output makes sense.** For each English query, the top candidate should be the Japanese posting that semantically matches it, not one that happens to share a few characters. If the rankings look wrong, suspect the prefix first: a missing `query:` is the most common cause.

4. **Use the result.** Pick one of your queries and answer this question in your head (or in a comment in the script): if you were going to set a threshold above which you would call a candidate a "real match", what cosine value would you pick based on what you see? There is no right answer; the point is to start developing the intuition you will need in Chapter 7.

## Common pitfalls

1. **Forgetting the `query:` prefix on one side of a comparison.** Symptoms: similarity scores look noticeably worse than they should, often by 0.05 to 0.15. There is no error. Fix: prefix every string going into e5, always, for this project.

2. **Mixing MiniLM and e5 vectors in the same comparison.** Symptoms: nonsense scores (vectors are different dimensions, so this will actually error; but if you had two same-dim models from different training schemes, it would silently fail). Fix: always pair vectors with the model that produced them. In Chapter 3 the provider abstraction will make this enforceable.

3. **Expecting low scores for unrelated text with e5.** Symptoms: you compare two unrelated Japanese sentences and get 0.78, and assume the model is broken. Fix: this is normal for e5. Its similarity floor sits high. What matters is the gap between matched and unmatched pairs, not the absolute number.

4. **Slow first run.** Symptoms: the multilingual model takes a minute or two to download on first use. Fix: nothing, just wait. After the first run the model loads from cache in seconds.

5. **Looking at a Japanese string and seeing boxes or `???`.** Symptoms: your terminal is not rendering Japanese characters. Fix: the model does not care about your terminal font; the bytes are correct. If you want to read the output, switch to a terminal that supports CJK (most modern ones do, including the default macOS Terminal, Windows Terminal, and any modern Linux terminal). Or open the script's output in a text editor that does.

6. **Confusing the prefix for a special token.** Symptoms: you try to find documentation about a `[QUERY]` token, or worry that the prefix gets stripped somewhere. Fix: it is literal text, six characters and a space, that the tokenizer treats like any other input. The model learned its meaning during training.

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge for Exercise 1</summary>

You need two models, one shared list of sentences, and two separate matrices of scores. The MiniLM matrix is built from raw sentences (no prefix). The e5 matrix is built from the same sentences with `"query: "` prepended to each. The same `cosine(a, b)` function works for both, because cosine similarity does not care which model produced the vectors, only that the two vectors come from the *same* model.

The visual proof comes from looking at the cells where row and column belong to a known bilingual pair. In the e5 matrix those cells should be visibly higher than the surrounding cells. In the MiniLM matrix they will not.

</details>

<details>
<summary>Hint 2 — Approach and pseudocode for both exercises</summary>

Exercise 1:

```
load MiniLM
load e5

sentences = [en_A, ja_A, en_B, ja_B, en_C, ja_C]
sentences_e5 = ["query: " + s for s in sentences]

emb_mini = MiniLM.encode(sentences)
emb_e5 = e5.encode(sentences_e5)

print "MiniLM matrix"
for i in range(N):
    for j in range(N):
        print cosine(emb_mini[i], emb_mini[j])

print "e5 matrix"
for i in range(N):
    for j in range(N):
        print cosine(emb_e5[i], emb_e5[j])
```

Exercise 2:

```
load e5

queries = [...]            # English
candidates = [...]         # Japanese
queries_pref = ["query: " + q for q in queries]
candidates_pref = ["query: " + c for c in candidates]

q_vecs = e5.encode(queries_pref)        # shape (Nq, 768)
c_vecs = e5.encode(candidates_pref)     # shape (Nc, 768)

define top_k(query_vec, candidate_vecs, k):
    scores = candidate_vecs @ query_vec        # shape (Nc,)
    pairs = list of (index, score)
    sort pairs by score descending
    return first k pairs

for each query:
    print query
    for index, score in top_k(...):
        print score, candidates[index]
```

</details>

<details>
<summary>Hint 3 — Code skeleton for Exercise 1</summary>

```python
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_matrix(name: str, embeddings: np.ndarray, labels: list[str]) -> None:
    """Print a labeled N by N pairwise cosine similarity matrix."""
    print(f"\n=== {name} ===")
    print("       ", end="")
    for j in range(len(labels)):
        print(f"  s{j}   ", end="")
    print()
    for i in range(len(labels)):
        print(f"  s{i}   ", end="")
        for j in range(len(labels)):
            print(f" {cosine(embeddings[i], embeddings[j]):.3f}", end=" ")
        print()


def main() -> None:
    """Compare MiniLM and multilingual-e5-base on bilingual sentence pairs."""
    sentences = [
        "Senior backend engineer with Python experience",
        "Pythonの経験を持つシニアバックエンドエンジニア",
        "I love hiking on the weekends",
        "週末にハイキングをするのが大好きです",
        "The new train line opened last month",
        "新しい鉄道路線が先月開業しました",
    ]
    sentences_e5 = [f"query: {s}" for s in sentences]

    mini = SentenceTransformer("all-MiniLM-L6-v2")
    e5 = SentenceTransformer("intfloat/multilingual-e5-base")

    emb_mini = mini.encode(sentences)
    emb_e5 = e5.encode(sentences_e5)

    print(f"e5 shape: {emb_e5.shape}")
    print(f"e5 norm of first vector: {np.linalg.norm(emb_e5[0]):.4f}")

    print_matrix("MiniLM (English-only)", emb_mini, sentences)
    print_matrix("multilingual-e5-base", emb_e5, sentences)


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary>Hint 4 — Code skeleton for Exercise 2</summary>

```python
import numpy as np
from sentence_transformers import SentenceTransformer


def top_k(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Return the top-k (index, score) pairs by cosine similarity, descending.

    Assumes both query_vec and rows of candidate_vecs are unit-normalized,
    so cosine similarity equals the dot product.
    """
    scores = candidate_vecs @ query_vec
    indexed = [(i, float(s)) for i, s in enumerate(scores)]
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    return indexed[:k]


def main() -> None:
    """Cross-language nearest-neighbor lookup using multilingual-e5-base."""
    queries = [
        "React frontend developer",
        "data engineer with Python and SQL",
        "DevOps engineer comfortable with Kubernetes",
    ]
    candidates = [
        "フロントエンドエンジニア募集 React TypeScript 経験者歓迎",
        "データエンジニア Python SQL BigQuery 経験必須",
        "インフラエンジニア AWS Kubernetes Docker 運用経験",
        "営業職 BtoB 法人営業 経験3年以上",
        "プロダクトマネージャー toC アプリ 経験3年以上",
        "バックエンドエンジニア Go gRPC マイクロサービス",
        "デザイナー Figma UI/UX 経験者",
        "QAエンジニア 自動テスト Playwright 経験",
    ]

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    q_vecs = model.encode([f"query: {q}" for q in queries])
    c_vecs = model.encode([f"query: {c}" for c in candidates])

    for i, query in enumerate(queries):
        print(f"\nquery: {query}")
        for idx, score in top_k(q_vecs[i], c_vecs, k=3):
            print(f"  {score:.3f}  {candidates[idx]}")


if __name__ == "__main__":
    main()
```

If you copy these, change at least the sentences and candidates so the exercise has been yours.

</details>

## Further reading

- The E5 paper, "Multilingual E5 Text Embeddings: A Technical Report": <https://arxiv.org/abs/2402.05672>
- The `intfloat/multilingual-e5-base` model card, which explicitly documents the prefix convention: <https://huggingface.co/intfloat/multilingual-e5-base>
- MTEB leaderboard filtered to multilingual models, useful for seeing where e5 sits relative to alternatives: <https://huggingface.co/spaces/mteb/leaderboard>
- "How to use multilingual embeddings", a short blog post by the sentence-transformers maintainers: <https://www.sbert.net/examples/training/multilingual/README.html>

## Checkpoint

Before moving to Chapter 3, you should have:

- [ ] `scripts/learn_multilingual.py` created, run, and producing two side-by-side similarity matrices that visibly differ in how they handle Japanese
- [ ] `scripts/multilingual_lookup.py` created, run, and printing sensible top-3 Japanese matches for each English query
- [ ] Confirmed `intfloat/multilingual-e5-base` returns vectors of shape `(N, 768)` and norm ~1.0
- [ ] Muscle memory for prefixing every e5 input with `query:` (or `passage:` when retrieving documents for a query, but you have not used that case yet)
- [ ] A working sense that e5's similarity floor sits higher than MiniLM's (often 0.7+ even for unrelated pairs) and that this is calibration, not a bug
- [ ] Code committed to your repo
