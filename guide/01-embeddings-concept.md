# Chapter 1: Embeddings, Concept and First Use

## What we're building in this chapter

A throwaway exploration script that loads a pre-trained sentence embedding model, converts a handful of English sentences into vectors, and computes the cosine similarity between each pair. The script does not become part of the service. It exists so you can see the math do something real, with your own sentences, on your own machine, before you wire it into FastAPI in later chapters.

## Why this matters

The entire project is built on one idea: that you can turn a piece of text into a list of numbers in such a way that texts with similar meaning produce similar lists, and you can measure that similarity with a single number. Every feature in the chapter plan (cross-language matching, skill canonicalization, 志望動機 coverage) is a different way of squeezing value out of that idea.

If you do not internalize what an embedding is and how similarity is measured, the later chapters will feel like incantations. After this chapter, when Chapter 4 asks you to store a 384-dimensional vector in pgvector, you will know exactly what is being stored and why 384 numbers are enough to compare two sentences meaningfully.

This chapter is also the one where you confront the disk and memory cost of modern ML libraries for the first time. Installing `sentence-transformers` pulls in PyTorch and downloads a model. It takes a few minutes and a few hundred megabytes. Knowing this now prevents confusion later.

## Concepts

### What is an embedding

An embedding is a fixed-length list of floating-point numbers that represents the meaning of a piece of text. The function that produces it is a neural network that has been trained, by reading huge amounts of text, to place semantically similar inputs near each other in the output space.

Pick the model `all-MiniLM-L6-v2` (the one you will use in this chapter). Feed it the string `"I love writing code"`. It returns a list of 384 floats, something like:

```
[-0.043, 0.012, -0.085, 0.137, ..., 0.021]  # 384 numbers
```

That list has no meaning to a human reading the individual numbers. None of the 384 positions corresponds to "is about coding" or "is positive sentiment". The meaning is distributed across all 384 axes at once, and only emerges when you compare two embeddings to each other.

Two key properties:

- **Fixed size, regardless of input length.** A one-word input and a 200-word input both produce a 384-float vector. This is what makes embeddings useful for indexing and comparison: every text is the same shape.
- **The size is a property of the model, not the text.** MiniLM produces 384 floats. OpenAI's `text-embedding-3-small` produces 1536. `multilingual-e5-base` (the one you will use in Chapter 2) produces 768. You cannot directly compare embeddings from different models. They live in different vector spaces.

### Vector spaces

A vector space is just a coordinate system. A 2D vector space has two axes (x, y) and every point in it is identified by two numbers. A 384-dimensional vector space has 384 axes and every point is identified by 384 numbers. You cannot visualize 384 dimensions, but the math of distance, direction, and angle generalizes without trouble.

In a trained embedding model, each of those 384 axes encodes some learned aspect of meaning. Nobody designed what axis 17 represents. It emerged from training. The result is that two pieces of text with similar meaning end up near each other in this 384-dimensional cloud, even though no human ever told the model "these belong together".

The mental picture to carry into the next chapters: imagine every sentence in the world as a dot floating in a huge dim space. Cooking sentences cluster in one region, code sentences in another, complaints about commutes in a third. To ask "how similar are these two sentences" is to ask "how close are these two dots".

### Comparing vectors: cosine similarity

There are two obvious ways to measure "how close are two points in a space". Euclidean distance is the straight-line distance between them. Cosine similarity is the cosine of the angle between the two vectors drawn from the origin to the points.

Cosine similarity is defined as:

```
cos_sim(A, B) = (A · B) / (|A| * |B|)
```

Where `A · B` is the dot product (multiply matching components, sum) and `|A|` is the L2 norm (square each component, sum, take square root).

The value ranges from -1 to 1:

- `1.0` means the vectors point in exactly the same direction. Maximum similarity.
- `0.0` means they are perpendicular. No relationship.
- `-1.0` means they point in opposite directions. Maximum dissimilarity.

In practice with sentence embeddings you almost never see negative values. The whole cloud sits in roughly one corner of the space, so scores cluster between about 0.1 (unrelated) and 0.95 (near-paraphrase). Knowing this range matters when you tune thresholds in Chapter 7.

### Why not Euclidean distance

Two embeddings of the same meaning can have different magnitudes (different vector lengths) for reasons that have nothing to do with what the text means. A longer sentence, a more confident prediction, or a quirk of the model's training can produce a vector with a larger norm. Euclidean distance is sensitive to magnitude. Cosine is not. It only cares about direction.

Another way to say it: cosine similarity normalizes for length so you compare *what* the vectors point at, not *how far* they reach. For text meaning, direction is what carries the signal.

There is one important shortcut: if your embeddings are already normalized to unit length (`|A| = |B| = 1`), then cosine similarity is just the dot product. Many models, including the ones you will use, return normalized vectors. The MiniLM model in this chapter does. You can verify this by computing the norm and checking it is very close to 1.0.

## The tools we're using

### `sentence-transformers`

- What it is: a Python library for loading and using sentence embedding models, built on top of Hugging Face's `transformers`.
- What it does for us: gives us a one-line API to download a pre-trained model and turn any string into a vector.
- Install: `pip install sentence-transformers`. This will also install `torch` (PyTorch), which is a large download (a few hundred MB on disk for the CPU-only wheel). Be patient.
- Docs: <https://www.sbert.net/>
- Key class and method:
  - `SentenceTransformer("model-name")` downloads the model the first time, caches it in `~/.cache/huggingface/`, and returns a model object. Subsequent calls with the same name load from cache and take seconds.
  - `model.encode("a string")` returns a NumPy array of shape `(384,)` for MiniLM. Pass a list of strings and you get a `(N, 384)` array.

### `numpy`

- What it is: the standard Python library for numeric arrays. You may have brushed past it in CS50P; it is the foundation of nearly every ML library in Python.
- What it does for us: holds the embedding vectors and gives us fast vectorized math for similarity calculations.
- Install: comes transitively with `sentence-transformers`. You do not need to add it to `requirements.txt` separately for this chapter, though many projects pin it explicitly. Up to you.
- Docs: <https://numpy.org/doc/stable/>
- Functions we will use:
  - `np.dot(a, b)` computes the dot product of two arrays.
  - `np.linalg.norm(a)` computes the L2 norm of an array (the square root of the sum of squared elements).
  - Indexing and basic arithmetic on arrays work elementwise. `a + b`, `a * 2`, etc. all do what you would intuitively expect.

### The model: `all-MiniLM-L6-v2`

- What it is: a small, fast English-only sentence embedding model trained by the sentence-transformers project. 6 transformer layers (hence L6), about 22 million parameters, 384-dimensional output.
- Size on disk: ~80MB.
- Why this one for Chapter 1: it downloads in under a minute, runs on a laptop without a GPU, and is the canonical "first embedding model" tutorial choice. You will swap it for the multilingual `intfloat/multilingual-e5-base` in Chapter 2, where you will see what an English-only model does when it is fed Japanese.
- Hugging Face page: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

## How it fits together

The data flow for the script you are about to write:

```
sentence strings (Python list)
        |
        v
SentenceTransformer("all-MiniLM-L6-v2").encode(...)
        |
        v
NumPy array of shape (N, 384)
        |
        v
for each pair (i, j):
    cos_sim = dot(emb[i], emb[j]) / (norm(emb[i]) * norm(emb[j]))
        |
        v
print N x N similarity matrix
```

The first run downloads the model. Every run after that loads from local cache and is fast. The script does not touch FastAPI, the `src/` package, or the database. It is intentionally separate: this chapter is about understanding the primitive, not building the service. Chapter 3 will start integrating embeddings into the real codebase under a provider abstraction.

## Code examples

### Example 1: Load a model and embed one string

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("I love writing code")

print(type(vector))    # <class 'numpy.ndarray'>
print(vector.shape)    # (384,)
print(vector.dtype)    # float32
print(vector[:5])      # the first five components
```

The first time you run this, you will see download progress bars for the model files. The second time, the model loads from `~/.cache/huggingface/` in a second or two.

### Example 2: Embed a batch in one call

```python
sentences = [
    "I love writing code",
    "Programming is fun",
    "The sky is blue today",
]
embeddings = model.encode(sentences)
print(embeddings.shape)   # (3, 384)
```

Passing a list to `encode` is more efficient than calling `encode` three times. The model batches the work internally. This matters when you embed dozens or hundreds of CV bullets at once in later chapters.

### Example 3: Cosine similarity by hand

```python
import numpy as np

a = embeddings[0]   # "I love writing code"
b = embeddings[1]   # "Programming is fun"
c = embeddings[2]   # "The sky is blue today"

def cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

print(cosine(a, b))   # ~0.55, both about coding/work
print(cosine(a, c))   # ~0.10, unrelated topics
print(cosine(b, c))   # ~0.08, unrelated topics
```

Notice that the "similar" pair scores around 0.55, not 0.99. Real-world embedding similarities rarely approach 1.0 unless the texts are near-identical paraphrases. Calibrating your expectations here is part of the chapter.

### Example 4: The built-in helper

```python
from sentence_transformers.util import cos_sim

scores = cos_sim(embeddings, embeddings)
print(scores.shape)   # torch.Size([3, 3])
print(scores)
```

`cos_sim` accepts a matrix and returns a matrix of pairwise similarities. The result is a `torch.Tensor` rather than a NumPy array, but it indexes and prints the same way for our purposes. Use this once you understand what it is computing under the hood. For Chapter 1, compute it by hand at least once so the formula is not magic.

## Your task

Write a small exploration script that loads the model, embeds several sentences, and prints a pairwise cosine similarity matrix.

1. **Add `sentence-transformers` to `requirements.txt`** at a pinned version of your choosing (look up the current version on PyPI). Reinstall with `pip install -r requirements.txt`. Expect a few minutes and a few hundred MB of downloads.

2. **Create `scripts/learn_embeddings.py`.** This file is exploratory. It is not part of the FastAPI service. You may delete or rewrite it later without affecting the project.

3. **Inside that script:**
   - Load `all-MiniLM-L6-v2` using `SentenceTransformer`.
   - Define a list of at least five sentences. Include some that are semantically similar (two sentences about programming, two about food, etc.) and some that are unrelated. Mix it up so the result is interesting.
   - Embed them all in a single `encode` call.
   - Write a `cosine(a, b)` function with type hints and a docstring that computes cosine similarity from two NumPy arrays using `np.dot` and `np.linalg.norm`. Do not use the `cos_sim` helper for this function. The point is to compute it yourself once.
   - Compute the full N by N pairwise similarity matrix and print it in a readable form. A simple nested loop with `print(f"{cosine(emb[i], emb[j]):.3f}", end=" ")` is fine. Label rows and columns with the sentence index or a short preview of the sentence.

4. **Run it twice.** The first run downloads the model. The second run should be fast. Confirm both work.

5. **Look at the matrix and answer two questions for yourself.** Do not write the answers down, just notice:
   - Which pair has the highest similarity? Does that match what you would expect from reading the sentences?
   - What is the typical range of scores for unrelated sentences? Is it really near zero, or does the floor sit higher?

6. **Verify the embeddings are unit-normalized.** Print `np.linalg.norm(embeddings[0])` and confirm it is very close to 1.0. This is a property of `all-MiniLM-L6-v2` specifically. Knowing whether your model produces normalized vectors will matter later when you optimize similarity calculations.

You do not need to commit this script as a meaningful milestone, but it is fine to commit it with a message like `chore: chapter 1 embedding exploration`.

## Common pitfalls

1. **The first run looks frozen.** It is downloading PyTorch dependencies (if you skipped installing them earlier) and the model weights. Expect a few minutes. If it actually freezes, check your network. The cache lives in `~/.cache/huggingface/`.

2. **`pip install sentence-transformers` is slow and big.** Several hundred MB on disk, mostly PyTorch. This is normal. If you are on a constrained machine, accept the disk hit; CPU-only PyTorch is what makes the model run without a GPU.

3. **Calling `model.encode("text")` returns a 1D array, but `model.encode(["text"])` returns a 2D array of shape `(1, 384)`.** If your similarity function expects 1D arrays and you accidentally pass 2D, you will get a confusing matrix shape or an error. Use `.shape` to check.

4. **Expecting "similar" sentences to score above 0.9.** They usually do not. Sentences about the same broad topic score in the 0.4 to 0.7 range with this model. Near-paraphrases score in the 0.7 to 0.9 range. Identical texts score 1.0. Calibrate accordingly.

5. **Mixing embeddings from different models or different normalization schemes.** A vector from MiniLM and a vector from OpenAI are not comparable, even if both have similar-looking floats. Always know which model produced a given embedding. This becomes a real concern in Chapter 3 when you build the provider abstraction.

6. **Using `float()` on the result.** `np.dot` and friends return NumPy scalar types like `numpy.float32`, which sometimes confuse downstream code (especially JSON serialization). Wrapping the result in `float(...)` returns a plain Python float and avoids surprises. This will matter when you serialize similarity scores in API responses.

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge</summary>

You need three things: a model object, a NumPy array of embeddings, and a function that takes two 1D arrays and returns a single float. The dot product of two normalized vectors is the cosine similarity. The dot product of two non-normalized vectors needs to be divided by the product of their norms. NumPy has both `np.dot` and `np.linalg.norm` ready for you.

For the matrix, you have N embeddings and want an N by N grid where cell `(i, j)` is the similarity between embedding `i` and embedding `j`. Two nested loops over `range(N)`.

</details>

<details>
<summary>Hint 2 — Approach and pseudocode</summary>

```
import SentenceTransformer
import numpy

define sentences (a list of strings)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

define cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print header row of column indices

for i in range(len(sentences)):
    print row label
    for j in range(len(sentences)):
        score = cosine(embeddings[i], embeddings[j])
        print score formatted to 3 decimal places
    newline

verify embeddings[0] has norm ~1.0
```

</details>

<details>
<summary>Hint 3 — Code skeleton</summary>

```python
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    """Embed several sentences and print a pairwise similarity matrix."""
    sentences = [
        "I love writing code",
        "Programming is my favorite hobby",
        "I make pasta for dinner",
        "Italian food is delicious",
        "The weather is nice today",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    print(f"shape: {embeddings.shape}")
    print(f"norm of first embedding: {np.linalg.norm(embeddings[0]):.4f}")
    print()

    # column header
    print("       ", end="")
    for j in range(len(sentences)):
        print(f"  s{j}  ", end="")
    print()

    # rows
    for i in range(len(sentences)):
        print(f"  s{i}   ", end="")
        for j in range(len(sentences)):
            print(f" {cosine(embeddings[i], embeddings[j]):.3f}", end=" ")
        print()


if __name__ == "__main__":
    main()
```

If you copy this, change at least the sentences so the exercise has been yours.

</details>

## Further reading

- Sentence Transformers documentation, the "Quickstart" page: <https://www.sbert.net/docs/quickstart.html>
- The original Word2Vec intuition, explained without heavy math: <https://jalammar.github.io/illustrated-word2vec/>
- 3Blue1Brown on word embeddings, 25 minutes well spent if you have not seen it: <https://www.youtube.com/watch?v=wjZofJX0v4M>
- The MTEB leaderboard, which ranks embedding models on standard benchmarks. Worth a glance to see where `all-MiniLM-L6-v2`, `multilingual-e5-base`, and OpenAI's models sit relative to each other: <https://huggingface.co/spaces/mteb/leaderboard>

## Checkpoint

Before moving to Chapter 2, you should have:

- [ ] `sentence-transformers` added to `requirements.txt` and installed
- [ ] `scripts/learn_embeddings.py` created and runs without error
- [ ] A printed N by N cosine similarity matrix for your chosen sentences
- [ ] A `cosine(a, b)` function written by hand using `np.dot` and `np.linalg.norm`
- [ ] Confirmed that `all-MiniLM-L6-v2` returns vectors of shape `(384,)` and norm ~1.0
- [ ] A working intuition for the typical similarity range (about 0.1 for unrelated sentences, 0.5 to 0.7 for same-topic sentences, 0.8+ for near-paraphrases)
- [ ] A clear mental model of why cosine and not Euclidean
