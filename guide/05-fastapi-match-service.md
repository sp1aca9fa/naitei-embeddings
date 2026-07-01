# Chapter 5: Building the FastAPI Service (the `/match` endpoint, in memory)

## What we're building in this chapter

A `POST /match` endpoint that takes a CV (a list of English bullets) and a job description (a block of text), and returns three things: a similarity matrix (every CV bullet scored against every JD requirement sentence), the coverage gaps (requirements that no bullet matches well), and the top matches per requirement. It runs entirely in memory this chapter. No database yet, that is Chapter 6. By the end you can open `/docs`, fill in a CV and a JD, and watch cross-language matching come back as JSON.

## Why this matters

This is Feature A, the core of the product, and the first time the embedding math you built becomes a real API contract that Naitei can call over HTTP. It is also your first proper FastAPI endpoint with a typed request and response. Everything after this (storage, auth, the second feature) hangs off the same FastAPI + Pydantic backbone, so the patterns here repeat for the rest of the project.

If it helps anchor it to what you already know: FastAPI is Express with typed, validated request/response bodies and free API docs. Pydantic is your DTO / zod layer, it defines the shape of data crossing the boundary and validates it for you. A FastAPI route is an Express handler; a Pydantic model is the schema you would have hand-written a validator for.

## Concepts

### The match problem is a matrix

You have N CV bullets and M JD requirement sentences. Score every bullet against every requirement and you get an N-by-M grid of cosine similarities:

```
                 req0    req1    req2   <- JD requirement sentences
        bullet0 [ 0.82    0.31    0.44 ]
        bullet1 [ 0.20    0.79    0.15 ]
        bullet2 [ 0.10    0.12    0.71 ]
           ^ CV bullets
```

Everything the endpoint returns is derived from this grid:

- **Top-N matches for a requirement** = look down that requirement's column, take the N highest cells. For `req0` above, bullet0 (0.82) is the best match.
- **Coverage gap** = a requirement whose *best* cell is still below a threshold. If the threshold is 0.7, every requirement above is covered. If `req1`'s column topped out at 0.4, it would be a gap: nothing in the CV addresses it well.

So the whole feature is: build the grid, then read it two ways (down columns for matches, column-maxima for gaps).

### Cosine similarity is just a dot product here

From Chapter 1, cosine similarity between vectors `a` and `b` is:

```
cos(a, b) = (a . b) / (||a|| * ||b||)
```

The denominator normalizes for length. But your providers already return **unit-length** vectors (mock divides by its norm, huggingface passes `normalize_embeddings=True`, OpenAI returns unit vectors). When `||a|| = ||b|| = 1`, the denominator is 1 and cosine similarity *is* the dot product:

```
cos(a, b) = a . b     (when a and b are unit vectors)
```

That is a big simplification. It means the entire N-by-M grid is one matrix multiply: stack your bullet vectors into a matrix `B` of shape `(N, d)` and your requirement vectors into `R` of shape `(M, d)`, and

```
matrix = B @ R.T        # shape (N, M)
```

gives you every pairwise similarity at once. NumPy does the whole grid in one fast operation instead of a Python double loop. (`@` is Python's matrix-multiply operator; `.T` transposes `R` from `(M, d)` to `(d, M)` so the shapes line up to produce `(N, M)`.)

### Sentence splitting, and why not `.split(".")`

The CV arrives as a list of bullets already, one string per bullet. But the JD arrives as one blob of text, and you need it broken into individual requirement sentences to score each one. The naive `jd_text.split(".")` breaks immediately:

- It splits on decimals ("3.5 years" becomes two fragments) and abbreviations ("e.g.", "etc.").
- Japanese does not end sentences with a period at all. It uses `。`, and a Japanese JD split on `.` stays one giant fragment.

Since your whole point is bilingual matching, you need a real sentence segmenter that understands both. `pysbd` (Python Sentence Boundary Disambiguation) handles many languages including Japanese, and you tell it which language it is looking at. The alternative, `nltk`, is heavier and weaker on Japanese, so use `pysbd`.

### Pydantic models are the API contract

Pydantic defines the shape of data at the boundary. You declare a class with typed fields; Pydantic parses incoming JSON into it, validates types and constraints, and rejects bad input before your code runs. FastAPI wires it in both directions: a Pydantic parameter becomes the request body (validated automatically, a bad body returns HTTP 422 without you writing a line), and a `response_model` shapes and documents what you send back. Both show up in the auto-generated docs at `/docs`.

This is the piece with the most new Python for you, so the tools section below has runnable examples. The mental model: it is a dataclass that validates itself and knows how to convert to and from JSON.

### `def` vs `async def` (a FastAPI choice worth understanding)

FastAPI lets you write a route as either `def handler(...)` or `async def handler(...)`. The difference matters:

- `async def` runs on the event loop. You must only `await` non-blocking work inside it. If you call something blocking (like embedding a batch, or a `psycopg2` query), you freeze the entire server for every user until it finishes.
- Plain `def` tells FastAPI to run your handler in a worker thread, so blocking work is fine and does not stall the event loop.

Your embedding calls (sentence-transformers, and later `psycopg2`) are blocking and CPU-bound. So write these endpoints as plain `def`. That is the correct, safe default here, and it is also less to think about. Only reach for `async def` when you are doing genuinely async I/O with libraries built for it. We are not, so `def` it is.

## The tools we're using

### FastAPI (request bodies and response models)

- What it is: the web framework serving your API. You met it in Chapter 0 for `/health`.
- What it does for us: turns a Pydantic class into a validated request body, shapes the response, and generates `/docs` for free.
- Install: already in `requirements.txt`.
- Docs: https://fastapi.tiangolo.com/tutorial/body/
- Key pieces, with a runnable example:
  - `@app.post("/path", response_model=OutModel)` declares a POST route and the response shape.
  - A parameter typed as a Pydantic model becomes the JSON request body.

  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI()

  class EchoIn(BaseModel):
      message: str
      times: int = 1        # default if the client omits it

  class EchoOut(BaseModel):
      repeated: list[str]

  @app.post("/echo", response_model=EchoOut)
  def echo(body: EchoIn) -> EchoOut:
      return EchoOut(repeated=[body.message] * body.times)
  ```

  Run `uvicorn ...:app --reload`, then:
  ```
  curl -X POST localhost:8000/echo \
    -H "Content-Type: application/json" \
    -d '{"message": "hi", "times": 3}'
  ```
  Output:
  ```json
  {"repeated": ["hi", "hi", "hi"]}
  ```
  Open http://localhost:8000/docs and the `/echo` endpoint is there with a "Try it out" form, generated from the models.

### Pydantic v2 (data models and validation)

- What it is: a data-validation library. FastAPI is built on it.
- What it does for us: defines and validates the `/match` request and response shapes.
- Install: comes with FastAPI.
- Docs: https://docs.pydantic.dev/latest/
- Key pieces, with a runnable example:
  - `BaseModel` subclass with typed fields; instantiating validates.
  - `Field(...)` adds constraints and defaults (`ge` = greater-or-equal, `le` = less-or-equal).
  - Models nest: a field can be another `BaseModel`.
  - `model_dump()` converts an instance to a plain dict (this is the v2 name; the old v1 `.dict()` is gone).

  ```python
  from pydantic import BaseModel, Field

  class Score(BaseModel):
      label: str
      value: float = Field(ge=0, le=1)     # must be within 0..1

  class Result(BaseModel):
      scores: list[Score]                  # a nested model list

  r = Result(scores=[Score(label="react", value=0.82)])
  print(r.model_dump())
  # {'scores': [{'label': 'react', 'value': 0.82}]}

  Score(label="bad", value=1.5)
  # raises pydantic.ValidationError: value: Input should be less than or equal to 1
  ```

  In FastAPI you rarely call `model_dump()` yourself for responses, returning the model (or a matching dict) is enough, but it is handy in scripts and tests.

### pysbd (sentence segmentation)

- What it is: a multilingual sentence boundary splitter.
- What it does for us: turns the JD blob into a clean list of requirement sentences, in English or Japanese.
- Install: `pip install pysbd`, then add `pysbd` to `requirements.txt`.
- Docs: https://github.com/nipunsadvilkar/pySBD
- Key pieces, with a runnable example:
  - `pysbd.Segmenter(language="en", clean=False)` builds a segmenter for one language.
  - `.segment(text)` returns a list of sentences.

  ```python
  import pysbd

  seg_en = pysbd.Segmenter(language="en", clean=False)
  print(seg_en.segment("We need React. 3.5 years exp. Strong TypeScript."))
  # ['We need React. ', '3.5 years exp. ', 'Strong TypeScript.']

  seg_ja = pysbd.Segmenter(language="ja", clean=False)
  print(seg_ja.segment("Reactの経験が必要です。TypeScriptも歓迎。"))
  # ['Reactの経験が必要です。', 'TypeScriptも歓迎。']
  ```
  Note "3.5" stayed intact, and the Japanese split on `。`. A `str.split(".")` would have failed both.

### NumPy (the similarity matrix and ranking)

- What it is: the array/math library your vectors already come back as.
- What it does for us: the one-shot similarity matrix and the ranking of matches.
- Install: comes with sentence-transformers.
- Docs: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
- Key pieces, with a runnable example:
  - `A @ B.T` matrix-multiplies to produce all pairwise dot products.
  - `matrix.max(axis=0)` / `matrix.argmax(axis=0)` reduce down each column.
  - `np.argsort(col)[::-1][:n]` gives the indices of the top n values, highest first.
  - `.tolist()` / `float(...)` convert NumPy numbers to plain Python (needed for JSON).

  ```python
  import numpy as np

  bullets = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)          # (2, 3), unit rows
  reqs    = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)  # (3, 3)

  matrix = bullets @ reqs.T     # (2 bullets, 3 requirements)
  print(matrix)
  # [[1. 0. 0.]
  #  [0. 0. 1.]]

  print(matrix.max(axis=0))     # best score per requirement -> [1. 0. 1.]
  print(matrix.argmax(axis=0))  # best bullet index per requirement -> [0 0 1]

  col0 = matrix[:, 0]           # requirement 0's column
  print(np.argsort(col0)[::-1][:2])  # top-2 bullet indices, best first -> [0 1]
  ```
  Reading `axis=0`: it collapses the rows (the bullets), leaving one value per column (per requirement). That is exactly "best bullet for each requirement."

## How it fits together

```
POST /match  {cv_bullets: [...], jd_text: "...", jd_language: "en"}
      |
      v
  split jd_text into requirement sentences        (pysbd)
      |
      v
  provider = get_provider()                        (same provider for both sides!)
  bullet_vecs = provider.embed(cv_bullets)         -> (N, d) unit rows
  req_vecs    = provider.embed(requirements)       -> (M, d) unit rows
      |
      v
  matrix = bullet_vecs @ req_vecs.T                -> (N, M) cosine grid   (NumPy)
      |
      +--> per requirement column: max score  -> coverage gap if < threshold
      +--> per requirement column: top-n bullets by score -> top matches
      |
      v
  MatchResponse(matrix=..., coverage_gaps=..., requirements=...)   (Pydantic)
      |
      v
  JSON out
```

Two things to hold onto. First, both sides must be embedded by the **same provider instance**, so the vectors share a space and the dot products mean something (the `model_name` lesson from Chapter 4, live again). Second, the endpoint is plain `def`, because `embed` blocks.

A note on structure, not for this chapter: as you add more endpoints you will split them into an `APIRouter` under `src/routes/` (the layout in PROJECT.md) instead of piling them into `main.py`. That is Express's `Router` idea. With a single new endpoint it is not worth it yet, so keep `/match` in `main.py` for now; we will introduce the router split when a second endpoint makes `main.py` crowded.

## Code examples

These are toy demonstrations on throwaway shapes, not the `/match` solution. They show the moving parts so you can assemble the real thing yourself.

**A nested Pydantic request/response pair:**
```python
from pydantic import BaseModel

class OrderIn(BaseModel):
    items: list[str]
    note: str = ""

class OrderOut(BaseModel):
    count: int
    items_upper: list[str]

o = OrderIn(items=["a", "b"])
print(OrderOut(count=len(o.items), items_upper=[i.upper() for i in o.items]).model_dump())
# {'count': 2, 'items_upper': ['A', 'B']}
```

**A toy "best match per column" over a fake grid (no embeddings):**
```python
import numpy as np

grid = np.array([
    [0.9, 0.1, 0.5],
    [0.2, 0.8, 0.4],
])
threshold = 0.7
best_per_col = grid.max(axis=0)                 # [0.9, 0.8, 0.5]
gaps = [j for j, s in enumerate(best_per_col) if s < threshold]
print(best_per_col, "gaps:", gaps)              # [0.9 0.8 0.5] gaps: [2]
```
Column 2's best is 0.5, below 0.7, so it is a gap. This is the coverage-gap logic on a fake grid; in the real thing the grid comes from `embed`.

**Splitting then counting sentences:**
```python
import pysbd
seg = pysbd.Segmenter(language="en", clean=False)
sents = seg.segment("Build APIs. Mentor juniors. Ship weekly.")
print(len(sents), sents)
# 3 ['Build APIs. ', 'Mentor juniors. ', 'Ship weekly.']
```

## Your tasks

You are building Feature A's endpoint in three steps: the contract, the logic, then a small extension. Exercises 1 and 2 are required; Exercise 3 is optional to complete but there for you.

### Exercise 1: Replication — the schema and a stubbed endpoint

Define the request and response contract with Pydantic, and wire a `POST /match` route in `src/main.py` that returns a **stub** (hard-coded or echoed values) so you can see the shape working in `/docs` before writing any real logic.

- Create a `MatchRequest` model with at least: `cv_bullets: list[str]`, `jd_text: str`, and `jd_language: str = "en"` (default, used later for the segmenter).
- Design the response. It must be able to carry: the similarity matrix, the coverage gaps, and the top matches per requirement. Use nested models rather than one flat blob. A reasonable shape (your call on names): a `MatchResponse` with `requirement_sentences: list[str]`, `cv_bullets: list[str]`, `matrix: list[list[float]]`, `coverage_gaps: list[str]`, and `requirements: list[RequirementMatch]`, where each `RequirementMatch` has the requirement text and its top matches (each a bullet plus score).
- Add `@app.post("/match", response_model=MatchResponse)` as a plain `def`, returning a stub built from the request (for example, echo the bullets back and return an empty matrix). No embeddings yet.
- Verify: `uvicorn src.main:app --reload`, open `/docs`, use "Try it out" on `/match`, and confirm the request/response shapes render and a stub response comes back. A `curl -X POST` works too.

**Pitfalls for this exercise:**
- Forgetting `-H "Content-Type: application/json"` on `curl` gives a 422; the body is not parsed as JSON without it. `/docs` sets it for you.
- Pydantic v2: to convert a model to a dict use `model_dump()`, not the removed v1 `.dict()`. For responses you can just return the model instance.
- Nested fields must themselves be `BaseModel` subclasses (a plain class or dict will not validate or document properly).

<details>
<summary>Stuck? Hints (click to expand)</summary>

- Conceptual nudge: sketch the JSON you would want back from a match, then make one `BaseModel` per nested object in that JSON. The response is a tree of models, not one flat model.
- Approach: define `BulletScore(bullet, score)`, then `RequirementMatch(requirement, top_matches: list[BulletScore])`, then `MatchResponse(...)` holding the lists. The endpoint returns a `MatchResponse(...)` built from stub data.
- Skeleton:
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI()

  class MatchRequest(BaseModel):
      cv_bullets: list[str]
      jd_text: str
      jd_language: str = "en"

  class BulletScore(BaseModel):
      bullet: str
      score: float

  class RequirementMatch(BaseModel):
      requirement: str
      top_matches: list[BulletScore]

  class MatchResponse(BaseModel):
      requirement_sentences: list[str]
      cv_bullets: list[str]
      matrix: list[list[float]]
      coverage_gaps: list[str]
      requirements: list[RequirementMatch]

  @app.post("/match", response_model=MatchResponse)
  def match(req: MatchRequest) -> MatchResponse:
      # stub: echo back, empty analysis
      return MatchResponse(
          requirement_sentences=[],
          cv_bullets=req.cv_bullets,
          matrix=[],
          coverage_gaps=[],
          requirements=[],
      )
  ```
  Keep `/health` where it is.

</details>

### Exercise 2: Application — the real match logic

Fill in the endpoint so it actually matches. Write a function (in `main.py` for now, or a small helper module if you prefer) that does the flow from "How it fits together," and call it from the route.

- Split `req.jd_text` into requirement sentences with a `pysbd.Segmenter(language=req.jd_language, clean=False)`.
- Get one provider with `get_provider()` and embed both `req.cv_bullets` and the requirement sentences with it. Remember `embed` takes a list and returns a 2D array of unit rows.
- Build the similarity matrix with a single matrix multiply.
- For each requirement (column): compute its best score, mark it a coverage gap if the best is below a threshold (hard-code `0.7` for now), and collect its top-N bullets (hard-code `N = 3`).
- Return everything in the `MatchResponse`. Convert NumPy numbers to plain Python (`float(...)`, `.tolist()`) so they serialize.
- Drive it: run with `EMBEDDING_PROVIDER=huggingface` and a small CV + JD (mix in a Japanese requirement). React/TypeScript bullets should rank above unrelated ones, and a Japanese requirement should still match an English bullet about the same skill. Mock will give near-random rankings by design, so use it only to check the plumbing, not the quality.

**Pitfalls for this exercise:**
- `embed` already returns 2D; pass the whole list, do not loop and stack single strings yourself, and do not wrap a lone string.
- Shape errors: `bullet_vecs @ req_vecs.T` is `(N, d) @ (d, M) = (N, M)`. Forget the `.T` and NumPy raises a dimension-mismatch error.
- `axis=0` collapses bullets (rows) to give one value per requirement (column). If your gaps look transposed, you used the wrong axis.
- NumPy floats (`float32`) can trip JSON serialization through Pydantic; wrap scalars in `float()` and matrices with `.tolist()`.
- Same-provider rule: embed bullets and requirements with the *same* provider instance, or the cosine scores are meaningless (different models, different spaces).
- A Japanese JD needs `jd_language="ja"`, or `pysbd` will not split on `。` and you get one giant "requirement."

<details>
<summary>Stuck? Hints (click to expand)</summary>

- Conceptual nudge: you already have `embed` and you learned the matrix is one multiply. The only new logic is reading columns: max for gaps, sorted indices for top-N.
- Approach (pseudocode):
  ```
  requirements = Segmenter(language).segment(jd_text)
  provider = get_provider()
  B = provider.embed(cv_bullets)          # (N, d)
  R = provider.embed(requirements)        # (M, d)
  matrix = B @ R.T                        # (N, M)
  for j, requirement in enumerate(requirements):
      col = matrix[:, j]
      best = col.max()
      if best < threshold: gaps.append(requirement)
      top_idx = argsort(col)[::-1][:n]
      top_matches = [(cv_bullets[i], float(col[i])) for i in top_idx]
  ```
- Skeleton:
  ```python
  import numpy as np
  import pysbd
  from src.providers import get_provider

  THRESHOLD = 0.7
  TOP_N = 3

  def run_match(cv_bullets, jd_text, jd_language):
      seg = pysbd.Segmenter(language=jd_language, clean=False)
      requirements = [s.strip() for s in seg.segment(jd_text)]

      provider = get_provider()
      bullet_vecs = provider.embed(cv_bullets)
      req_vecs = provider.embed(requirements)
      matrix = bullet_vecs @ req_vecs.T          # (N, M)

      coverage_gaps = []
      requirement_matches = []
      for j, requirement in enumerate(requirements):
          col = matrix[:, j]
          if col.max() < THRESHOLD:
              coverage_gaps.append(requirement)
          top_idx = np.argsort(col)[::-1][:TOP_N]
          top = [BulletScore(bullet=cv_bullets[i], score=float(col[i])) for i in top_idx]
          requirement_matches.append(RequirementMatch(requirement=requirement, top_matches=top))

      return MatchResponse(
          requirement_sentences=requirements,
          cv_bullets=cv_bullets,
          matrix=matrix.tolist(),
          coverage_gaps=coverage_gaps,
          requirements=requirement_matches,
      )
  ```
  Then the route is `return run_match(req.cv_bullets, req.jd_text, req.jd_language)`.

</details>

### Exercise 3 (optional to complete): client-configurable threshold and top-N

Right now the threshold and N are hard-coded. Promote them to request fields with real validation, so the caller controls them and bad values are rejected automatically. This deepens Pydantic (the chapter's core tool) and stays on the endpoint you just built.

- Add to `MatchRequest`: `threshold: float = Field(0.7, ge=0, le=1)` and `top_n: int = Field(3, ge=1)`.
- Thread them through the match logic instead of the hard-coded constants.
- Confirm FastAPI rejects out-of-range input on its own: POST a body with `threshold: 1.5` or `top_n: 0` and you should get a `422` with a clear message, without you writing any validation code.

**Pitfalls for this exercise:**
- The default goes as the first positional arg to `Field`: `Field(0.7, ge=0, le=1)`. Putting the value in `default=` also works, but do not omit it, or the field becomes required.
- `0.7` is a placeholder threshold; the right value depends on the model and is genuinely tuned in Chapter 7. Do not agonize over it here.
- Do not hand-write range checks in the handler; the whole point is that `Field` plus FastAPI produce the 422 for you.

<details>
<summary>Stuck? Hints (click to expand)</summary>

- Conceptual nudge: a validated field is one line; the 422 is free. The only real work is passing two new values down instead of reading module constants.
- Approach: give `run_match` two more parameters (`threshold`, `top_n`) with the request's values, and delete the module-level `THRESHOLD` / `TOP_N`.
- Skeleton:
  ```python
  from pydantic import BaseModel, Field

  class MatchRequest(BaseModel):
      cv_bullets: list[str]
      jd_text: str
      jd_language: str = "en"
      threshold: float = Field(0.7, ge=0, le=1)
      top_n: int = Field(3, ge=1)
  ```
  Then `run_match(..., threshold=req.threshold, top_n=req.top_n)` and use those inside.

</details>

## Common pitfalls (chapter-wide)

- **`async def` with blocking work.** If you write the handler as `async def` and then call `embed` (or later `psycopg2`), you block the event loop and stall the whole server. Use plain `def` here; FastAPI runs it in a thread pool.
- **Pydantic v1 muscle memory.** Method names changed in v2: `model_dump()` not `.dict()`, `model_dump_json()` not `.json()`. Old tutorials will steer you wrong.

## Further reading

- FastAPI request body: https://fastapi.tiangolo.com/tutorial/body/
- FastAPI response model: https://fastapi.tiangolo.com/tutorial/response-model/
- Pydantic v2 fields and validation: https://docs.pydantic.dev/latest/concepts/fields/
- pySBD (why rule-based segmentation, language support): https://github.com/nipunsadvilkar/pySBD

## Checkpoint

Before moving to Chapter 6, you should have:
- [ ] `MatchRequest` and the nested `MatchResponse` models defined with Pydantic
- [ ] `POST /match` wired in `main.py` as a plain `def`, visible and testable at `/docs`
- [ ] Real match logic: sentence-split JD, embed both sides with one provider, similarity matrix via matrix multiply, coverage gaps, and top-N per requirement
- [ ] `pysbd` added to `requirements.txt` and used for splitting
- [ ] Verified meaningful cross-language ranking under `huggingface` (React bullet matching a Japanese React requirement, unrelated requirements showing as gaps)
- [ ] (optional) `threshold` and `top_n` as validated request fields returning 422 on bad input
- [ ] Understood why these handlers are `def`, not `async def`
- [ ] Code committed to your repo
