# Chapter 3: Provider Abstraction

## What we're building in this chapter

A new `src/providers/` package with an `EmbeddingProvider` abstract base class and three concrete implementations: `HuggingFaceProvider` (wraps `sentence-transformers` with `multilingual-e5-base`, the model you used in Chapter 2), `OpenAIProvider` (calls OpenAI's `text-embedding-3-small` over HTTPS), and `MockProvider` (deterministic fake vectors for tests, no network, no model download). Plus a small `src/config.py` for environment variable loading and a `get_provider()` factory so the rest of the service can ask for "the embedding provider" without knowing or caring which one is configured.

After this chapter, you can swap the entire backing model with a single environment variable and zero code changes. You also produce the first real piece of `src/` code in the project. The exploration scripts under `scripts/` from Chapters 1 and 2 stay as they are, but from here on the FastAPI service grows.

## Why this matters

Naitei already does this on the TypeScript side. It has a provider-agnostic AI layer that lets the same call site work against Claude, OpenAI, Gemini, or Ollama with one environment variable. The Python service is going to follow the same pattern for embeddings, for the same three reasons Naitei does:

1. **You will change your mind about the model.** Today `multilingual-e5-base` looks right. In six months a newer multilingual model will land on the MTEB leaderboard. You want to try it without rewriting the service.

2. **Tests must not call the real model.** Loading `multilingual-e5-base` takes 5 to 15 seconds and burns 280 MB of disk. Calling OpenAI's API costs money and requires network. A test suite that does either is a test suite you will not run often enough. The mock provider exists to make every test cheap.

3. **Production might want a different provider than local development.** During development you probably want the local HuggingFace model so you can work offline and not pay per call. In production, OpenAI may be the better choice once you start indexing thousands of jobs (its API is faster than running e5 on a small Fly.io box). Same code, different env var.

There is a fourth reason that matters specifically for this project: the abstraction is also where model-specific quirks live. The `query:` prefix you internalized in Chapter 2 is an e5 quirk. The rest of the service should not have to know about it. The `HuggingFaceProvider` will add the prefix internally, so when later chapters write `provider.embed(["React developer"])` they get a sensible vector back without sprinkling `f"query: {text}"` across the codebase. If you later switch to OpenAI, no prefix is needed, and the caller's code does not change. That is the abstraction earning its keep.

## Concepts

### Provider abstraction in general

A "provider" in this style is a small object that hides a specific external dependency (a model, a database, a third-party API) behind a uniform interface. The rest of your code calls methods on the interface and never touches the dependency directly. This is the same shape as a Rails service object, a TypeScript class implementing an `interface`, or a Go struct satisfying an interface implicitly. In Naitei's TypeScript code it shows up as something like:

```typescript
interface AIProvider {
  generate(prompt: string): Promise<string>;
}

class ClaudeProvider implements AIProvider { ... }
class OpenAIProvider implements AIProvider { ... }
```

Python's equivalent is the **abstract base class**. Python does not have an `interface` keyword, but the `abc` module in the standard library provides a way to declare that a class has methods which subclasses must implement.

### Python abstract base classes

The `abc` module gives you two pieces: the `ABC` base class (something to inherit from) and the `@abstractmethod` decorator (something to mark methods as required). When a class inherits from `ABC` and has at least one method marked `@abstractmethod`, Python refuses to let you instantiate it directly.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        """Return the sound this animal makes."""

class Dog(Animal):
    def speak(self) -> str:
        return "woof"

Animal()    # TypeError: Can't instantiate abstract class Animal with abstract method speak
Dog()       # works
Dog().speak()  # "woof"
```

The enforcement happens at instantiation time, not at class-definition time. If you define `Cat(Animal)` and forget to implement `speak`, Python lets the class definition succeed. The `TypeError` shows up the moment someone tries to call `Cat()`. That is enough to catch real mistakes without slowing development down.

Two small Python idioms worth absorbing:

- An abstract method can have an empty body or just a docstring. It does not have to `pass` or `raise NotImplementedError`. The `@abstractmethod` decorator is what makes it abstract; the body is just documentation.

- You can combine `@property` and `@abstractmethod`. The order matters: `@property` must be the outer decorator.

  ```python
  class Animal(ABC):
      @property
      @abstractmethod
      def legs(self) -> int:
          """How many legs this animal has."""
  ```

  Subclasses then implement it as a normal `@property`. This is how you require subclasses to expose a piece of data without dictating how they compute or store it.

### What our `EmbeddingProvider` interface should look like

Think about what the rest of the service will ask of an embedding provider. It will hand over a list of strings and want back a 2D array of vectors. It will sometimes want to know how many dimensions the vectors have (the database schema in Chapter 4 needs this to declare the `vector(n)` column type). It will sometimes want a stable name for the provider (for logging, and later for tagging stored embeddings with which model produced them).

That suggests three things on the interface:

- `embed(texts: list[str]) -> np.ndarray` — the main method. Takes a list, returns a `(len(texts), dimension)` numpy array of unit-normalized vectors.
- `dimension: int` — read-only property. The number of components in each output vector.
- `name: str` — read-only property. A stable identifier like `"intfloat/multilingual-e5-base"` or `"openai/text-embedding-3-small"` or `"mock"`.

Two implementation contracts the interface implies but doesn't enforce in code:

- All output vectors are **unit-normalized** (norm equal to 1.0 within float precision). This means cosine similarity reduces to a dot product, the same simplification you used in Chapter 2.
- The `embed` method is **batch-shaped, always**. Even for one string, you pass a list of length one and get back an array of shape `(1, dimension)`. No special case for a single input. This keeps callers simple.

Conventions like normalization are easy to forget. The chapter calls them out, you implement them in each provider, and Chapter 10's tests will verify them.

### Deterministic mocks: hash to seed

The mock provider's job is to look like an embedding provider without doing any of the work. It must:

1. Return vectors of the configured dimension, shaped `(len(texts), dimension)`.
2. Return unit-normalized vectors so cosine math behaves the same as with real providers.
3. Return the same vector for the same input string, every time, across processes. This is the property that makes tests reproducible.

The standard trick: hash the input text to a fixed integer, use that as the seed for a numpy random generator, and pull the vector from that generator.

```python
import hashlib
import numpy as np

def _seed_from_text(text: str) -> int:
    """Hash text to a 32-bit unsigned int suitable for seeding numpy."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

rng = np.random.default_rng(_seed_from_text("React developer"))
vec = rng.standard_normal(768)
vec /= np.linalg.norm(vec)
```

A few things worth understanding:

- **Why `hashlib.sha256` and not Python's built-in `hash()`.** Python's `hash()` is salted per process by default, for security reasons (it prevents certain denial-of-service attacks against dicts). That salting means `hash("foo")` returns a different value in every new process. Useless for reproducible mocks. `hashlib` hashes are deterministic across processes and platforms.

- **Why we slice to 4 bytes (32 bits).** `np.random.default_rng()` accepts a seed up to `2**63 - 1`, but you do not need that much. 32 bits gives 4 billion distinct seeds, more than enough that no two of your test inputs will collide.

- **Why `standard_normal`.** It draws from a normal distribution centered at zero, which is what real embedding models effectively produce after training. Uniform random vectors would have a different statistical shape and would not be a good stand-in. The downstream cosine math does not actually care, but consistency with real-world distributions is a small piece of self-discipline that costs nothing.

- **Why we normalize.** Without it, the mock vectors would have norms scattered around `sqrt(dimension)`, and cosine math via simple dot product would silently break. The whole project assumes unit-normalized vectors. Every provider must honor that.

### Environment-driven configuration

Hardcoding which provider to use ("just call `HuggingFaceProvider()` in main.py") would force a code edit every time you want to swap. The standard fix is a single environment variable, read at startup, that tells the factory which provider to construct.

For local development, environment variables live in a `.env` file at the project root. The `python-dotenv` library reads that file and populates `os.environ`, so your code can do `os.getenv("EMBEDDING_PROVIDER")` and not care whether the value came from a real shell env var or from `.env`. In production on Fly.io or Render, the platform injects real env vars and `.env` is irrelevant.

The `.env` file is gitignored (you already have `.env` in `.gitignore` from Chapter 0). The committed `.env.example` documents what variables exist so future-you (and the deployment platform) knows what to set.

For this chapter you will need three variables:

- `EMBEDDING_PROVIDER` — one of `mock`, `huggingface`, `openai`. Default: `mock`. The default matters; it means a fresh clone of the repo with no `.env` still runs.
- `OPENAI_API_KEY` — required only when `EMBEDDING_PROVIDER=openai`.
- `HUGGINGFACE_MODEL` — optional override of the local model name. Default: `intfloat/multilingual-e5-base`.

You may also want `MOCK_DIMENSION` to match whatever real provider you would swap the mock for in tests. Default to 768 (e5's dimension).

### Where the e5 prefix lives

Chapter 2 drilled in that every input to `multilingual-e5-base` must be prefixed with `query:` for the symmetric similarity case this project lives in. That rule should not become a chant the rest of the codebase has to remember. It belongs inside the `HuggingFaceProvider`. The provider's `embed(texts)` method will accept raw strings, prepend `query:` internally, call the model, and return the resulting vectors.

The OpenAI provider does not need any prefix. Its `embed(texts)` accepts the same raw strings and just forwards them. The mock provider does not need any prefix either; it just hashes whatever you give it.

The shape of this design is: **each provider owns its model-specific transformations**. Callers know nothing about prefixes, normalization kwargs, API request bodies, or random seeds. They know about `embed(texts)`, `dimension`, and `name`. That is the entire interface.

### A note on async (we are not using it yet)

The OpenAI Python SDK supports both sync and async clients. FastAPI handlers can be sync or async. For this chapter, all providers are synchronous. The reason is pedagogical: async is a separate concept that deserves its own chapter, and the embedding latency in this project is not the bottleneck. When we wire `/match` together in Chapter 5 we will reconsider, but for now `def embed(...)` is a normal blocking function and that is fine.

## The tools we're using

### `abc` (Python standard library)

- What it is: a stdlib module that provides infrastructure for declaring abstract base classes.
- What it does for us: lets us define `EmbeddingProvider` as a class that cannot be instantiated directly and that forces subclasses to implement specific methods.
- Install: none, it is built in.
- Docs: <https://docs.python.org/3/library/abc.html>
- Key items we'll use:
  - `ABC` — base class to inherit from when defining an abstract class.
  - `@abstractmethod` — decorator that marks a method as required by subclasses.

### `python-dotenv`

- What it is: a small library that reads a `.env` file into the process's environment.
- What it does for us: lets us put `EMBEDDING_PROVIDER=mock` in a gitignored `.env` and access it via `os.getenv` without writing parsing code.
- Install: `pip install python-dotenv`, then pin in `requirements.txt`.
- Docs: <https://github.com/theskumar/python-dotenv>
- Key items we'll use:
  - `load_dotenv(path)` — call this once at startup to populate `os.environ` from a file.

### `openai` (the Python SDK)

- What it is: the official OpenAI Python client library.
- What it does for us: gives us a typed client for calling OpenAI's embeddings endpoint (and many others we will not use).
- Install: `pip install openai`, then pin in `requirements.txt`.
- Docs: <https://github.com/openai/openai-python>
- Embeddings docs: <https://platform.openai.com/docs/guides/embeddings>
- Key items we'll use:
  - `OpenAI(api_key=...)` — construct the client.
  - `client.embeddings.create(model=..., input=[...])` — synchronous call. Returns an object whose `.data[i].embedding` is a Python list of floats.

### `hashlib` (Python standard library)

- What it is: stdlib module providing common cryptographic hash functions.
- What it does for us: the `sha256` hash gives us deterministic-across-processes integer seeds for the mock provider.
- Install: none, built in.
- Key items we'll use:
  - `hashlib.sha256(b"...").digest()` — returns the 32-byte hash of the input bytes.

### `numpy` and `sentence-transformers` (carried over)

You already use both. No new methods this chapter, but worth knowing one new keyword argument:

- `SentenceTransformer.encode(texts, normalize_embeddings=True)` — passing `normalize_embeddings=True` divides each output vector by its L2 norm before returning. This makes the unit-normalized contract explicit at the call site rather than implicit in the model's behavior. Use it inside `HuggingFaceProvider`.

## How it fits together

```
                          .env file (gitignored)
                          +--------------------------+
                          | EMBEDDING_PROVIDER=mock  |
                          | OPENAI_API_KEY=sk-...    |
                          | HUGGINGFACE_MODEL=...    |
                          +-------------+------------+
                                        |
                                        v
                          +--------------------------+
                          |    src/config.py         |
                          |  load_dotenv()           |
                          |  read variables          |
                          +-------------+------------+
                                        |
                                        v
                          +--------------------------+
                          | src/providers/__init__   |
                          |   get_provider() ------+ |
                          +-------------+----------|-+
                                        |          |
                          +-------------+----------|------------+
                          |             |          |            |
                          v             v          v            v
                  +-------+---+  +------+----+  +--+--------+
                  | Mock      |  | HF        |  | OpenAI    |
                  | Provider  |  | Provider  |  | Provider  |
                  +-----+-----+  +-----+-----+  +-----+-----+
                        |              |              |
                        |   all three implement       |
                        |   the same EmbeddingProvider|
                        |   abstract base class       |
                        |                             |
                        +--------------+--------------+
                                       |
                                       v
                            future caller (Chapter 5+)
                            provider.embed(["..."]) -> np.ndarray
                            provider.dimension -> int
                            provider.name -> str
```

The dashed contract is the abstract base class. The boxes below it are the three concrete classes. The caller above the dashed contract never imports the concrete classes directly. It only ever calls `get_provider()` and uses the returned object through its abstract interface.

In Chapter 4 the database schema picks a vector dimension based on whichever provider you have configured. In Chapter 5 the `/match` endpoint becomes the first real caller of `provider.embed(...)`. In Chapter 10 the tests construct a `MockProvider` directly to keep the test suite hermetic. All of those follow naturally from the abstraction you build here.

## Code examples

### Example 1: A minimal abstract base class

```python
from abc import ABC, abstractmethod

class Greeter(ABC):
    """Says hello in some language."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Two-letter code identifying the language."""

    @abstractmethod
    def greet(self, name: str) -> str:
        """Return a greeting addressed to name."""


class EnglishGreeter(Greeter):
    @property
    def language(self) -> str:
        return "en"

    def greet(self, name: str) -> str:
        return f"Hello, {name}"


class JapaneseGreeter(Greeter):
    @property
    def language(self) -> str:
        return "ja"

    def greet(self, name: str) -> str:
        return f"{name}さん、こんにちは"


# Greeter()  -> TypeError, cannot instantiate abstract class
EnglishGreeter().greet("Ada")        # "Hello, Ada"
JapaneseGreeter().greet("Ada")       # "Adaさん、こんにちは"
```

Notice that the two concrete classes share no code yet behave interchangeably anywhere a `Greeter` is expected. That substitutability is the entire point.

### Example 2: Deterministic mock vector

```python
import hashlib
import numpy as np

def mock_vector(text: str, dimension: int) -> np.ndarray:
    """Return a unit-normalized fake vector that depends only on text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dimension).astype(np.float32)
    return v / np.linalg.norm(v)


a = mock_vector("React developer", 768)
b = mock_vector("React developer", 768)
c = mock_vector("Vue developer", 768)

print(np.allclose(a, b))       # True
print(np.allclose(a, c))       # False
print(float(np.linalg.norm(a)))  # 1.0 (within float precision)
```

The first assertion is the property that makes mocks useful in tests: same input, same output. The second confirms different inputs give different outputs. The third confirms the unit norm.

### Example 3: Loading environment variables

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

provider_name = os.getenv("EMBEDDING_PROVIDER", "mock")
print(provider_name)  # "mock" by default, or whatever .env sets
```

The path calculation walks up from `src/config.py` (this file) to the project root where `.env` lives. Using `pathlib` rather than `os.path` is one of the project's style rules.

`os.getenv("X", "default")` returns the default if `X` is not set. Always provide a default for non-secret config, so a fresh clone runs without a `.env` file.

### Example 4: A factory dispatching by name

```python
def get_provider(name: str) -> Greeter:
    """Return a Greeter instance for the given language code."""
    if name == "en":
        return EnglishGreeter()
    if name == "ja":
        return JapaneseGreeter()
    raise ValueError(f"Unknown greeter: {name}")


greeter = get_provider(os.getenv("LANG_CODE", "en"))
print(greeter.greet("Raphael"))
```

A factory function is just a function that decides which concrete class to instantiate based on input. There is no magic here. The real `get_provider()` you write will look essentially like this, with the three embedding providers in place of greeters and with a little extra validation for the OpenAI key.

## Your tasks

### Exercise 1: Replication — build the providers package

**1. Update `requirements.txt`.** Add three packages: `openai`, `python-dotenv`, and `pytest`. Pin to current versions. Then install:

```bash
pip install -r requirements.txt
```

If you do not have an OpenAI account or do not want to spend money, that is fine. You still install the package (the `OpenAIProvider` file must import it), but you will not actually run the OpenAI provider end-to-end this chapter. Run with `EMBEDDING_PROVIDER=mock` or `EMBEDDING_PROVIDER=huggingface` instead.

**2. Update `.env.example` and create `.env`.** Add three lines to `.env.example`:

```
EMBEDDING_PROVIDER=mock
OPENAI_API_KEY=
HUGGINGFACE_MODEL=intfloat/multilingual-e5-base
```

Copy `.env.example` to `.env` (which is gitignored) and fill in `OPENAI_API_KEY` if you have one. Otherwise leave it blank and keep `EMBEDDING_PROVIDER=mock`.

**3. Create `src/config.py`.** It should:

- Call `load_dotenv()` pointing at the project's `.env` file.
- Expose module-level constants for `EMBEDDING_PROVIDER`, `OPENAI_API_KEY`, `HUGGINGFACE_MODEL`, and `MOCK_DIMENSION`.
- Provide sensible defaults for everything except `OPENAI_API_KEY` (which is only required when the OpenAI provider is selected, so it can be `None` otherwise).

**4. Create `src/providers/__init__.py`** as the package entry point. For now it can be empty or simply re-export names; you will add `get_provider()` to it in step 8.

**5. Create `src/providers/base.py`** with the `EmbeddingProvider` abstract base class. It should declare:

- An abstract property `name: str`.
- An abstract property `dimension: int`.
- An abstract method `embed(texts: list[str]) -> np.ndarray`.

Use docstrings to document the contracts (unit-normalized output, batch-shaped, what `name` is for).

**6. Create `src/providers/mock.py`** implementing `MockProvider(EmbeddingProvider)`:

- Constructor takes a `dimension: int` and stores it.
- `name` returns `"mock"`.
- `dimension` returns the stored value.
- `embed(texts)` hashes each text with `hashlib.sha256`, seeds a `numpy.random.default_rng`, draws a `standard_normal(dimension)` vector, normalizes it, and stacks the results into a `(len(texts), dimension)` array.

The factoring tip from the chapter: write a small private helper `_seed_from_text(text: str) -> int` so the seeding logic is testable in isolation.

**7. Create `src/providers/huggingface.py`** implementing `HuggingFaceProvider(EmbeddingProvider)`:

- Constructor takes a `model_name: str` with a default of `"intfloat/multilingual-e5-base"`. Loads a `SentenceTransformer` and stores it. Reads and stores the model's output dimension via `model.get_sentence_embedding_dimension()`.
- `name` returns the model name.
- `dimension` returns the stored dimension.
- `embed(texts)` prepends `"query: "` to each input, calls `model.encode(prefixed, normalize_embeddings=True)`, and returns the resulting array.

**8. Create `src/providers/openai.py`** implementing `OpenAIProvider(EmbeddingProvider)`:

- Constructor takes an `api_key: str`. Constructs `OpenAI(api_key=api_key)` and stores the client.
- `name` returns `"openai/text-embedding-3-small"`.
- `dimension` returns `1536` (a class-level constant is fine).
- `embed(texts)` calls `self._client.embeddings.create(model="text-embedding-3-small", input=texts)`, collects the embeddings from the response, and returns them as a numpy float32 array. `text-embedding-3-small` is already normalized on OpenAI's end; you do not need to re-normalize.

A note on imports inside this file. You will write `from openai import OpenAI` at the top. That resolves to the installed `openai` package, not your local file, because Python uses absolute imports by default. The local file is reachable only as `src.providers.openai`. If you ever see a "cannot import name OpenAI from openai" error, it is almost certainly an editor doing the wrong thing on your behalf; the import is correct.

**9. Write `get_provider()` in `src/providers/__init__.py`.** It should:

- Read `config.EMBEDDING_PROVIDER`.
- Dispatch to the matching concrete class.
- For the OpenAI branch, raise `RuntimeError` with a clear message if `config.OPENAI_API_KEY` is unset.
- For an unknown name, raise `ValueError`.

Signature: `def get_provider() -> EmbeddingProvider:`. No arguments. The point is that callers do not need to think.

**10. Verify everything wires together.** Add a small script `scripts/try_provider.py` that:

- Imports `get_provider`.
- Calls it.
- Prints `provider.name` and `provider.dimension`.
- Calls `provider.embed(["React developer", "Reactエンジニア", "I made pasta"])`.
- Prints the shape of the result and the norm of the first row.
- For each pair of inputs, prints the cosine (which should reduce to dot product since vectors are normalized).

Run it three times, changing `EMBEDDING_PROVIDER` in `.env` between `mock`, `huggingface`, and (if you have a key) `openai`. The output shape and norms should be consistent with each provider's documented dimension and the unit-norm guarantee. The cosine between the English and Japanese strings should be high with `huggingface` and `openai` (this is what Chapter 2 already showed). With `mock` the cosines will be near zero, because random vectors in high dimensions are nearly orthogonal. That is correct behavior for a mock; it is not pretending to be a real model.

### Exercise 2: Application — pin down the mock with a test

This is the first real test in the project. The mock provider has a contract you wrote in code (deterministic, unit-normalized, correct dimension), and a test is the right way to lock that contract in.

**1. Create `tests/test_providers.py`.** Use pytest. You should not need any pytest plugins or fixtures yet, just `def test_*` functions.

**2. Write at least three tests:**

- `test_mock_is_deterministic` — construct two `MockProvider(dimension=64)` instances independently, embed the same list of strings on each, and assert the resulting arrays are equal with `np.array_equal` or `np.allclose`.

- `test_mock_vectors_are_unit_normalized` — embed a handful of strings, compute `np.linalg.norm(result, axis=1)`, and assert all norms are within `1e-5` of 1.0.

- `test_mock_dimension_is_respected` — construct `MockProvider(dimension=128)`, embed three strings, and assert the result shape is `(3, 128)`.

A fourth optional test, if you want practice: `test_mock_different_texts_give_different_vectors` — confirm that two different strings produce vectors that are not equal. This is the property that makes the mock useful in similarity tests.

**3. Run the suite.** From the project root:

```bash
pytest tests/test_providers.py -v
```

All tests should pass. If any fail, the failure is real: it means your mock provider does not actually have the property you thought you were enforcing. Fix the provider, not the test.

**4. One reflective question** (no code; answer in your head or in a comment). The mock provider produces near-zero cosines for any pair of inputs, because two independent random vectors in 768-dimensional space are almost always nearly orthogonal. This makes the mock useless for testing "does this CV bullet match this JD requirement", because there is no semantic signal. So why have it at all? What does the mock actually let you test? The answer should be in mind by the end of Chapter 5, when you see what kinds of failures the real `/match` endpoint can have.

## Common pitfalls

1. **`providers/openai.py` and the `openai` package name overlap.** Symptom: confused mental model, occasional auto-import nonsense from your editor. Fix: ignore the overlap. Python's absolute imports route `from openai import OpenAI` to the installed package. Your local file is only ever addressable as `src.providers.openai`. The collision is cosmetic.

2. **Forgetting to normalize mock vectors.** Symptom: norms come out around `sqrt(dimension)` rather than 1.0. The unit norm test catches it. Fix: divide by `np.linalg.norm(v)` before returning each row.

3. **Using `hash()` instead of `hashlib.sha256` for the mock seed.** Symptom: tests pass in one session and fail in the next, with no code changes. Fix: Python's built-in `hash()` is process-salted by default; never use it for anything that needs to persist or be reproducible across runs. `hashlib` is the right tool.

4. **Letting the e5 prefix leak into the caller.** Symptom: somewhere in `scripts/try_provider.py` or later in `/match`, you write `provider.embed([f"query: {bullet}"])`. Fix: do not. The prefix is the `HuggingFaceProvider`'s problem. Callers pass raw text. If you find yourself adding `query:` outside `huggingface.py`, the abstraction has sprung a leak.

5. **Dimensions disagreeing across providers in your tests.** Symptom: a test constructs `MockProvider(dimension=384)` because you copy-pasted from somewhere, but the rest of the project assumes 768. Fix: default the mock to whatever `config.MOCK_DIMENSION` says (default 768, matching e5). In Chapter 4 the database schema will pin a dimension and you will want test mocks to match it.

6. **Committing `.env`.** Symptom: your `OPENAI_API_KEY` ends up on GitHub. Fix: `.env` is in `.gitignore` already from Chapter 0. Verify with `git status` that `.env` is not staged. Only `.env.example` is ever committed.

7. **`openai` package not installed when you set `EMBEDDING_PROVIDER=openai`.** Symptom: `ImportError` at startup. Fix: keep the import at the top of `openai.py`. Even if you do not have a key, install the package. The mock and huggingface branches still work without an OpenAI key as long as you do not select that provider.

8. **`get_provider()` called many times and loading the HuggingFace model each time.** Symptom: every call to `get_provider()` takes 5 to 15 seconds because it reloads the model. Fix for this chapter: do not worry about it; `get_provider()` is only meant to be called once at startup. If you want to be safe, wrap the call site in a module-level singleton (`provider = get_provider()` at import time). The right fix in FastAPI is dependency injection via `Depends`, which we will cover when we wire `/match` together.

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge</summary>

The whole package is small. There are five files: `base.py` (one abstract class, three abstract members), `mock.py`, `huggingface.py`, `openai.py` (each one concrete class), and `__init__.py` (one factory function). Plus `config.py` outside the package (a few `os.getenv` calls).

If you find yourself writing more than ~30 lines per provider, you are probably adding things that do not belong yet. The provider is just a thin adapter. Caching, batching across the wire, retries, async, none of that lives here. Just the three abstract members and whatever the underlying library or SDK requires to fulfill them.

The factory is also small. Read one env var, branch on its value, construct the right class, return it. If your `get_provider()` is more than ~15 lines, look again.

</details>

<details>
<summary>Hint 2 — Approach and pseudocode</summary>

Pseudocode for the package layout:

```
src/config.py:
    load_dotenv(project_root / ".env")
    EMBEDDING_PROVIDER = getenv("EMBEDDING_PROVIDER", "mock")
    OPENAI_API_KEY = getenv("OPENAI_API_KEY")          # may be None
    HUGGINGFACE_MODEL = getenv("HUGGINGFACE_MODEL", "intfloat/multilingual-e5-base")
    MOCK_DIMENSION = int(getenv("MOCK_DIMENSION", "768"))

src/providers/base.py:
    class EmbeddingProvider(ABC):
        @property @abstractmethod name -> str
        @property @abstractmethod dimension -> int
        @abstractmethod embed(texts: list[str]) -> np.ndarray

src/providers/mock.py:
    class MockProvider(EmbeddingProvider):
        __init__(dimension): store dimension
        name -> "mock"
        dimension -> stored dimension
        embed(texts):
            for each text:
                seed = sha256(text)[:4] as int
                v = default_rng(seed).standard_normal(dimension)
                v /= norm(v)
            stack and return

src/providers/huggingface.py:
    class HuggingFaceProvider(EmbeddingProvider):
        __init__(model_name): load SentenceTransformer, read dim
        name -> model_name
        dimension -> stored dim
        embed(texts):
            prefixed = ["query: " + t for t in texts]
            return model.encode(prefixed, normalize_embeddings=True)

src/providers/openai.py:
    class OpenAIProvider(EmbeddingProvider):
        __init__(api_key): build OpenAI client
        name -> "openai/text-embedding-3-small"
        dimension -> 1536
        embed(texts):
            resp = client.embeddings.create(model=..., input=texts)
            return np.array([d.embedding for d in resp.data], dtype=float32)

src/providers/__init__.py:
    def get_provider() -> EmbeddingProvider:
        match config.EMBEDDING_PROVIDER:
            "mock"        -> MockProvider(config.MOCK_DIMENSION)
            "huggingface" -> HuggingFaceProvider(config.HUGGINGFACE_MODEL)
            "openai"      -> require key; OpenAIProvider(config.OPENAI_API_KEY)
            else          -> ValueError
```

Pseudocode for `tests/test_providers.py`:

```
from src.providers.mock import MockProvider

def test_mock_is_deterministic():
    a = MockProvider(64).embed(["foo", "bar"])
    b = MockProvider(64).embed(["foo", "bar"])
    assert np.array_equal(a, b)

def test_mock_vectors_are_unit_normalized():
    out = MockProvider(64).embed(["foo", "bar", "baz"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_mock_dimension_is_respected():
    out = MockProvider(128).embed(["a", "b", "c"])
    assert out.shape == (3, 128)
```

</details>

<details>
<summary>Hint 3 — Code skeleton for base.py and mock.py</summary>

`src/providers/base.py`:

```python
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """A source of unit-normalized embedding vectors for input text.

    Concrete providers wrap a specific model or API. Callers depend only
    on this interface; they never import concrete classes directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider. Used for logging and tagging."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of components in each output vector."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings.

        Returns an array of shape (len(texts), self.dimension). Each row is
        unit-normalized so cosine similarity reduces to a dot product.
        """
```

`src/providers/mock.py`:

```python
import hashlib

import numpy as np

from .base import EmbeddingProvider


def _seed_from_text(text: str) -> int:
    """Hash text to a 32-bit unsigned integer suitable for seeding numpy."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class MockProvider(EmbeddingProvider):
    """Deterministic fake embeddings. Same input always yields the same vector.

    Intended for tests. Does not produce semantically meaningful similarity.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dimension) array of unit-normalized vectors."""
        rows = []
        for text in texts:
            rng = np.random.default_rng(_seed_from_text(text))
            v = rng.standard_normal(self._dimension).astype(np.float32)
            v /= np.linalg.norm(v)
            rows.append(v)
        return np.stack(rows)
```

</details>

<details>
<summary>Hint 4 — Code skeleton for huggingface.py, openai.py, and the factory</summary>

`src/providers/huggingface.py`:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider


class HuggingFaceProvider(EmbeddingProvider):
    """Local embeddings via sentence-transformers.

    Default model is intfloat/multilingual-e5-base, which expects a
    "query: " prefix on inputs for the symmetric similarity case. This
    class adds the prefix internally so callers never have to think about it.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self._model = SentenceTransformer(model_name)
        self._name = model_name
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True)
```

`src/providers/openai.py`:

```python
import numpy as np
from openai import OpenAI

from .base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """Cloud embeddings via OpenAI's text-embedding-3-small.

    Outputs are already unit-normalized on OpenAI's end.
    """

    _MODEL = "text-embedding-3-small"
    _DIMENSION = 1536

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"openai/{self._MODEL}"

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._MODEL, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)
```

`src/providers/__init__.py`:

```python
from .. import config
from .base import EmbeddingProvider
from .huggingface import HuggingFaceProvider
from .mock import MockProvider
from .openai import OpenAIProvider

__all__ = [
    "EmbeddingProvider",
    "HuggingFaceProvider",
    "MockProvider",
    "OpenAIProvider",
    "get_provider",
]


def get_provider() -> EmbeddingProvider:
    """Return the embedding provider selected by EMBEDDING_PROVIDER in the environment."""
    name = config.EMBEDDING_PROVIDER
    if name == "mock":
        return MockProvider(dimension=config.MOCK_DIMENSION)
    if name == "huggingface":
        return HuggingFaceProvider(model_name=config.HUGGINGFACE_MODEL)
    if name == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIProvider(api_key=config.OPENAI_API_KEY)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {name}")
```

`src/config.py`:

```python
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "mock")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "intfloat/multilingual-e5-base")
MOCK_DIMENSION: int = int(os.getenv("MOCK_DIMENSION", "768"))
```

If you copy any of this, change names, formatting, or structure where they do not match how you already work. The exercise is to internalize the abstraction, not to reproduce one specific arrangement.

</details>

## Further reading

- The `abc` module reference: <https://docs.python.org/3/library/abc.html>
- Python's PEP 3119, which introduced ABCs and explains the rationale: <https://peps.python.org/pep-3119/>
- OpenAI's embeddings guide, especially the "Use cases" section: <https://platform.openai.com/docs/guides/embeddings>
- The 12-Factor App's section on config (the canonical argument for environment-variable-based configuration): <https://12factor.net/config>

## Checkpoint

Before moving to Chapter 4, you should have:

- [ ] `requirements.txt` updated with `openai`, `python-dotenv`, and `pytest`, and dependencies installed
- [ ] `.env.example` updated with the three new variables, and a local `.env` file (gitignored) populated for your machine
- [ ] `src/config.py` loading environment variables from `.env`
- [ ] `src/providers/base.py` defining the `EmbeddingProvider` abstract base class
- [ ] `src/providers/mock.py`, `src/providers/huggingface.py`, `src/providers/openai.py` each implementing the abstract class
- [ ] `src/providers/__init__.py` exposing `get_provider()` as the single entry point
- [ ] `scripts/try_provider.py` runnable under at least the `mock` and `huggingface` settings, printing sensible shapes, norms, and cosines
- [ ] `tests/test_providers.py` containing at least three passing tests that pin down the mock provider's determinism, normalization, and dimension contract
- [ ] An internalized sense that the `query:` prefix is the `HuggingFaceProvider`'s private concern and never leaks above the abstraction
- [ ] Code committed to your repo
