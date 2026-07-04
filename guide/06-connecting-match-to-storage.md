# Chapter 6: Connecting Match to Storage

## What we're building in this chapter

You will make `/match` stop re-embedding the same text on every request. Instead, before embedding a CV bullet or a JD requirement sentence, your code will check the database for an embedding of that exact text under the current provider, reuse it if found, and only call the provider (and store the result) the first time that text is ever seen. By the end, calling `/match` twice with the same CV and JD costs one embedding pass, not two.

## Why this matters

Chapter 5 built `/match` fully in memory: every request re-embeds every CV bullet and every JD requirement sentence from scratch, even if you sent the exact same CV five minutes ago. That is fine for a chapter about getting the logic right. It is not fine for a real service. Under the `huggingface` provider, embedding is a local model forward pass, so redundant calls cost latency. Under `openai`, redundant calls cost real money on every single request, for text you already paid to embed once. A user re-checking their match against the same job posting, or an interface that calls `/match` repeatedly as someone edits one bullet, would otherwise re-embed everything else that did not change.

Chapter 4 already gave you the storage half of this: three tables that hold text alongside its vector. What is missing is the *policy* that decides, before embedding anything, "have I seen this exact text before, under this exact model?" That policy is the "embed-once" pattern, and it is the piece that turns your database from a place embeddings get archived into a real cache that the running service depends on. This is also the last piece before Chapter 7 reuses the same pattern for skill canonicalization, and before Chapter 9's backfill script has to reason about which rows are safe to skip re-embedding.

## Concepts

### Content-addressed caching

The general problem: given a piece of text, quickly answer "do I already have an embedding for this?" without scanning every stored row and comparing strings character by character. The standard solution is to compute a short, fixed-size fingerprint of the text — a hash — and look up that fingerprint instead of the text itself. Two identical strings always hash to the same value; two different strings essentially never collide (a cryptographic hash like SHA-256 makes accidental collisions astronomically unlikely). This is called content-addressed caching: the cache key is derived entirely from the content, not from an external identifier like a database row number.

This is the same idea, applied one level differently, as what your `MockProvider` already does: it hashes input text to seed a random generator so the same text always yields the same fake vector. Here you hash text to *look up* a real stored vector instead of to seed randomness, but the underlying move, text in, deterministic fixed-size fingerprint out, is identical.

The direct payoff for "when to re-embed": you never have to write logic that detects "this CV bullet was edited, invalidate its cache entry." If the text changes by even one character, its hash changes, so it simply becomes a new, distinct cache key that has never been seen before and gets embedded fresh. The old row for the old wording is still sitting in the table, unreferenced, which is fine and cheap; you are not required to clean it up for this project's scale. Content-addressing turns cache invalidation, normally a hard problem, into something that falls out for free.

### Why the model name has to be part of the key

Two different providers embedding the identical string do not produce comparable vectors: different models, different dimensions, different coordinate systems entirely. If your cache key were text alone, switching `EMBEDDING_PROVIDER` from `mock` to `huggingface` and re-running `/match` would find the *old* mock vector for text it has "seen before" and hand it back as if it were a real embedding, silently corrupting your similarity matrix (or crashing on a dimension mismatch, if you are lucky enough for the dimensions to differ). The cache key has to be the pair (text, model name), not text alone. You already store `model_name` on every row from Chapter 4; this chapter is where that column starts pulling weight beyond bookkeeping.

### Schema evolution: adding a column and a uniqueness rule after the fact

Chapter 4's tables did not anticipate a hash-based lookup. Evolving a live schema is itself a normal, repeated event in real projects, not a one-time setup step: you add a column with `ALTER TABLE`, and if a column's values need to be unique (here, unique per model, since two different models can legitimately produce two different rows for the same text), you enforce that with a unique index over more than one column at once, a *compound* unique index. That is different from the single-column primary key you already have; a compound unique index says "no two rows may share this *combination* of values," while individual values in isolation can repeat freely.

### Read-then-write races, and why we are accepting them here

The natural way to implement "check cache, then write if missing" is exactly two steps in your Python code: a `SELECT`, then, if nothing came back, an `INSERT`. If two requests hit that gap at the same time, both can see a miss and both insert, giving you two rows for the same (text, model) pair, one now silently orphaned. Postgres has a real fix for this, `INSERT ... ON CONFLICT DO NOTHING`, which makes the insert itself a no-op if the row already exists, closing the race entirely. For this project, a single-user service with no concurrent traffic, the two-step check-then-insert is an acceptable simplification and it is what you will build. Know that `ON CONFLICT` exists and is the real answer once concurrency is a factor; it is called out again in the pitfalls below.

## The tools we're using

### `hashlib` (standard library)

- What it is: Python's standard-library module for cryptographic hash functions (SHA-256, MD5, and others).
- What we're trying to achieve with it: turn an arbitrary piece of text, a CV bullet or a JD requirement sentence of any length, into a short, fixed-length string that is safe and fast to store, index, and compare in the database.
- Why it's needed for this project specifically: this fixed-length fingerprint is the cache key described above. Without it you would have to index and compare on the raw `source_text` column directly, which is slower to index (text columns are variable-length and can be long) and does not change the fact that you still need *some* deterministic way to say "have I seen this before."
- Quick comparison with alternatives: you could index the text column itself instead of hashing it. It would work, but a `text` index over potentially long JD sentences is heavier than a fixed-length hash index, and you already have `hashlib` experience from `MockProvider`'s `_seed_from_text`, so reusing the same tool for a related purpose is the smaller step. MD5 would also technically work here (you are not defending against a malicious adversary, just avoiding accidental collisions), but SHA-256 is what you already used, so stay consistent.
- How it fits into our code: the hash of a bullet or requirement sentence is computed right before you would otherwise call `provider.embed`, and that hash is what gets passed to your new database lookup function and, on a miss, stored alongside the row you insert.
- Install: standard library, no install.
- Docs: <https://docs.python.org/3/library/hashlib.html>
- Key methods we'll use, with a runnable example:
  - `hashlib.sha256(data: bytes)` — creates a hash object from bytes (not `str` directly; text must be encoded first).
  - `.hexdigest()` — returns the hash as a lowercase hex string, safe to store in a `text` column and to compare with `==`.

```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

print(content_hash("Built REST APIs using FastAPI"))
print(content_hash("Built REST APIs using FastAPI"))
print(content_hash("built rest apis using fastapi"))
```

```
a3f1e9... (some 64-character hex string)
a3f1e9... (identical to the line above)
7c02b4... (different — case changed the input, so the hash changed too)
```

The last line matters: this hash is case-sensitive and whitespace-sensitive. That is a deliberate simplicity for this chapter, not an oversight; treat any normalization of text (trimming, lowercasing) as a separate decision you are not required to make here, and see the pitfalls section for what it would cost you if you skipped thinking about it entirely.

### NumPy: stacking individual vectors into a matrix

- What it is: you have used NumPy since Chapter 1 for vectors and matrix multiplication. The one operation you have not needed yet is building a 2D array out of several separately-obtained 1D vectors.
- What we're trying to achieve with it: up to now, `provider.embed(list_of_texts)` handed you an already-stacked 2D array in one call, because every text was embedded together in one batch. Once some vectors come from the database (cache hits) and others come fresh from the provider (cache misses), you are assembling the final `bullet_vecs` / `req_vecs` arrays one row at a time, in a loop, in the same order as the input list. You need a way to turn that growing list of 1D vectors back into the single 2D array the rest of `run_match`'s matrix math expects.
- Why it's needed for this project specifically: `bullet_vecs @ req_vecs.T` from Chapter 5 only works if `bullet_vecs` and `req_vecs` are proper 2D arrays with one row per bullet/requirement, in the same order the rest of the function assumes. Get the order wrong and the matrix's rows or columns no longer line up with the `cv_bullets` / `requirements` lists you're zipping them against.
- Quick comparison with alternatives: you could pre-allocate an empty array of the right shape and fill it in by index (`np.empty((n, dim))` then `arr[i] = vec`). That works too, but building a plain Python list of 1D vectors and converting it once at the end with `np.array(...)` is simpler to read and just as correct at this scale.
- How it fits into our code: this is the last step before the similarity matrix multiply, the seam where "a list of individually-fetched vectors" becomes "the 2D array Chapter 5's code already knows how to multiply."
- Docs: <https://numpy.org/doc/stable/reference/generated/numpy.array.html>
- Key usage, runnable:

```python
import numpy as np

vectors = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
]
matrix = np.array(vectors)
print(matrix.shape)
print(matrix)
```

```
(2, 3)
[[0.1 0.2 0.3]
 [0.4 0.5 0.6]]
```

Each 1D vector became one row, in the order it was appended to the list. `np.vstack(vectors)` does the same thing here and either is fine; `np.array` on a list of same-shape 1D arrays is enough for this case.

### SQL: evolving a schema and `ON CONFLICT`

- What it is: `ALTER TABLE` changes an existing table's structure without dropping it; `CREATE UNIQUE INDEX` on more than one column enforces uniqueness across the combination of those columns; `ON CONFLICT` is an `INSERT` clause that defines what to do when a row would violate a uniqueness constraint.
- What we're trying to achieve with it: add the new hash column to tables that already exist and have rows in them, and make "the same text under the same model" impossible to store twice by accident, at the database level rather than trusting your Python to always check first.
- Why it's needed for this project specifically: Chapter 4's tables predate this chapter's caching design, so the hash column has to be added after the fact, the normal shape of a real schema change. The compound uniqueness constraint is what makes the (text, model) pairing you decided on above an actual guarantee, not just a convention you hope your code respects.
- How it fits into our code: run these once, by hand, in the Supabase SQL editor, exactly like Chapter 4's initial schema. Your Python code in `src/db.py` then assumes the column and constraint already exist.
- Docs: <https://www.postgresql.org/docs/current/sql-altertable.html>, <https://www.postgresql.org/docs/current/sql-createindex.html>
- Key syntax, runnable in the SQL editor against a toy table:

```sql
CREATE TABLE demo_rows (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_hash text NOT NULL,
    model_name text NOT NULL
);

ALTER TABLE demo_rows ADD COLUMN note text;

CREATE UNIQUE INDEX ON demo_rows (content_hash, model_name);

INSERT INTO demo_rows (content_hash, model_name, note) VALUES ('abc', 'mock', 'first');
INSERT INTO demo_rows (content_hash, model_name, note) VALUES ('abc', 'mock', 'second')
    ON CONFLICT (content_hash, model_name) DO NOTHING;

SELECT * FROM demo_rows;
```

The second insert is silently skipped because `(content_hash, model_name)` already exists as `('abc', 'mock')`; only one row, with `note = 'first'`, remains. You are not required to use `ON CONFLICT` in your own code this chapter (the read-then-write approach is what you'll build, per the concept above), but seeing it work on a toy table means you recognize it later when concurrency actually matters.

## How it fits together

```
      run_match(cv_bullets, jd_text, ...)
                    |
        for each bullet / requirement:
                    |
                    v
        content_hash = sha256(text).hexdigest()
                    |
                    v
        db.get_embedding_by_hash(table, content_hash, provider.name)
                    |
          +---------+---------+
          | hit               | miss
          v                   v
    reuse stored vector   provider.embed([text])[0]
          |                   |
          |                   v
          |         db.insert_embedding(..., content_hash=...)
          |                   |
          +---------+---------+
                    |
                    v
        append vector to this side's list, in input order
                    |
                    v
        np.array(list) -> bullet_vecs / req_vecs (2D)
                    |
                    v
        bullet_vecs @ req_vecs.T   (unchanged from Chapter 5)
```

Both sides of `/match`, the CV bullets and the JD requirement sentences, go through the same check-then-embed-then-store step, just against different tables (`cv_bullet_embeddings` and `job_embeddings`). The matrix math and coverage-gap logic downstream of that do not change at all from Chapter 5; only where the vectors come from changes.

## Code examples

These examples work against a throwaway `demo_embeddings` table, not your real project tables, so you still write the real integration yourself.

### Example 1: a get-or-create function against a toy table

```sql
CREATE TABLE demo_embeddings (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_hash text NOT NULL,
    text_value text NOT NULL,
    embedding vector(3) NOT NULL,
    model_name text NOT NULL
);
CREATE UNIQUE INDEX ON demo_embeddings (content_hash, model_name);
```

```python
import hashlib
import numpy as np

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_or_create_demo_embedding(conn, text: str, model_name: str) -> np.ndarray:
    h = content_hash(text)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM demo_embeddings WHERE content_hash = %s AND model_name = %s",
            (h, model_name),
        )
        row = cur.fetchone()
        if row:
            return row[0]

    vec = np.array([0.1, 0.2, 0.3])  # stand-in for a real provider.embed call
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO demo_embeddings (content_hash, text_value, embedding, model_name) VALUES (%s, %s, %s, %s)",
            (h, text, vec, model_name),
        )
    conn.commit()
    return vec
```

Calling this twice with the same `text` and `model_name` hits the `SELECT` branch the second time and never reaches the `INSERT`. Calling it with the same `text` but a different `model_name` misses (because the compound key did not match) and creates a second row.

### Example 2: assembling a matrix from mixed hits and misses

```python
import numpy as np

texts = ["alpha", "beta", "gamma"]
cache = {"alpha": np.array([1.0, 0.0]), "gamma": np.array([0.0, 1.0])}  # "beta" is a miss

vectors = []
for t in texts:
    if t in cache:
        vectors.append(cache[t])
    else:
        vectors.append(np.array([0.5, 0.5]))  # stand-in for a fresh provider.embed call

matrix = np.array(vectors)
print(matrix.shape)
```

```
(3, 2)
```

Order is preserved because the loop walks `texts` in order and appends exactly one vector per text, whether it came from the cache or not.

## Your tasks

### Exercise 1 (Fixate): a get-or-create embedding function

Add the schema and the single reusable function this chapter's caching pattern depends on.

**What to do:**

1. In the Supabase SQL editor, add a `content_hash` text column to the tables `/match` actually uses (`cv_bullet_embeddings` and `job_embeddings`), and create a compound unique index over the hash column and the model-name column on each. Decide whether `content_hash` should be `NOT NULL` yourself, based on what you're about to always provide.
2. In `src/db.py`, write a function that, given a table name, a piece of text, and a model name, returns the stored vector if a matching (hash, model) row already exists, and otherwise returns nothing found. Base it on the same table-name-driven approach your existing `insert_embedding` / `get_embedding` already use, so it stays consistent with your Chapter 4 code rather than introducing a second style.
3. Update your existing insert function (or add a variant) so it also accepts and stores the content hash you computed, since a future lookup needs it there to find the row again.
4. Write a small helper (in `db.py`, or a new module if you prefer) that ties the two together: given a table name, a piece of text, a model name, and a way to produce a fresh embedding when needed, it returns a vector, reusing a stored one when available and creating one when not.

**How to verify it works:** write a short throwaway script that calls your get-or-create helper twice with the same text and table, and prints the row count in that table before and after (via a direct `SELECT count(*)`) to confirm the second call did not insert a new row. Then call it once more with the same text but a different value for `EMBEDDING_PROVIDER`, and confirm the count *did* go up by one.

**Pitfalls for this exercise:** forgetting to add the model name to the lookup query (you'll silently get cross-model hits); forgetting the unique index, which means nothing stops duplicate rows from piling up even though your Python thinks it's being careful; passing the hash as a `%s` value is fine (it's a value, not an identifier) but don't forget it has to be encoded to bytes before hashing, not hashed as a raw Python string.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** You already have all three tools this needs from earlier chapters and this chapter's tool sections: `hashlib` to turn text into a lookup key, a `SELECT ... WHERE` query shaped like the ones you already wrote in Chapter 4 (just with two conditions in the `WHERE` clause instead of one), and your existing insert function's pattern, extended with one more column. Write the hash function first and prove it in isolation (call it twice on the same string, confirm equal output) before touching the database at all. Then write the lookup query and prove it returns nothing on an empty table. Only then wire the "if nothing found, insert" branch on top, so you are never debugging more than one new piece at a time.

**Approach / pseudocode:**
```
to get or create an embedding for some text, under some model, in some table:
    compute a fingerprint of the text
    ask the table: is there a row with this fingerprint and this model?
    if yes: hand back its stored vector
    if no: produce a new vector, save it (with its fingerprint and model attached), hand back the new vector
```

**Code skeleton:**
```python
def get_embedding_by_hash(table_name: str, content_hash: str, model_name: str) -> np.ndarray | None:
    """Return the stored vector for this (hash, model) pair, or None if not found."""
    ...  # SELECT embedding FROM {table} WHERE content_hash = %s AND model_name = %s


def get_or_create_embedding(table_name: str, ref_id: str, text: str, model_name: str, embed_fn) -> np.ndarray:
    """Reuse a stored embedding for text if one exists, otherwise embed and store it."""
    h = content_hash(text)
    existing = get_embedding_by_hash(table_name, h, model_name)
    if existing is not None:
        return existing
    vec = embed_fn(text)
    insert_embedding(table_name, ref_id, text, vec, model_name, content_hash=h)
    return vec
```

`embed_fn` is left as a plug-in point on purpose: you decide whether `get_or_create_embedding` calls `provider.embed` directly or receives it as an argument.

</details>

### Exercise 2 (Apply): wire the cache into `/match`

Replace the bulk `provider.embed(...)` calls in `run_match` with your get-or-create helper, for both the CV bullets and the JD requirement sentences, without changing the shape of the response.

**What to do:**

1. In `run_match`, instead of calling `provider.embed` once on the whole list of CV bullets, loop over the bullets and get each one's vector through your Exercise 1 helper against `cv_bullet_embeddings`, collecting the results in the same order the bullets came in.
2. Do the same for the JD requirement sentences against `job_embeddings`.
3. Turn each collected list back into the 2D array the rest of the function expects, and confirm the matrix multiply still produces a matrix of the same shape as before.
4. Your get-or-create helper needs some identifier per row (Chapter 4's `ref_id`). Since `/match` requests don't carry a real Naitei job ID or CV version ID yet (that integration is still ahead), decide what to pass here yourself; think about what identifies a row well enough for now without pretending it's a real upstream ID.

**How to verify it works:** call `/match` twice in a row with the same CV bullets and JD text via `/docs`, and confirm (by checking row counts in the two tables via the SQL editor) that the second call added zero new rows while still returning the same matrix and coverage gaps as the first call.

**Pitfalls for this exercise:** losing the original order when collecting vectors in the loop breaks the alignment between `matrix` rows/columns and your `cv_bullets` / `requirements` lists, silently, with no error, so double check row and column ordering against known input rather than assuming it's right; mixing up which table belongs to which side; forgetting that this now makes one embed call per bullet/requirement on a cache miss instead of one batched call for all of them, which is slower on a first-ever request but is the tradeoff this caching model makes.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** Keep Chapter 5's `run_match` structure exactly as it is; you are only changing the two lines that currently call `provider.embed(cv_bullets)` and `provider.embed(requirements)` in bulk. Replace each with a loop that calls your Exercise 1 helper once per item and appends the result to a list, then convert that list to a 2D array the same way the tools section showed. Everything after that, the matrix multiply, the column loop, the response model, stays untouched. For the identifier question, look at what you already have on hand inside `run_match` for each piece of text, there isn't a real ID from upstream yet, so something derived from the text itself is a reasonable stand-in until Naitei integration exists.

**Approach / pseudocode:**
```
for the CV side:
    for each bullet, in order:
        get its vector via the cache helper, using some per-item identifier
        keep the vector in a list, same order as the bullets
    turn that list into a 2D array

do the same for the JD side, using the requirement sentences and the other table

everything from "build the similarity matrix" onward is unchanged from before
```

**Code skeleton:**
```python
def run_match(cv_bullets, jd_text, jd_language, threshold, top_n) -> MatchResponse:
    seg = pysbd.Segmenter(language=jd_language, clean=False)
    requirements = [s.strip() for s in seg.segment(jd_text)]

    provider = get_provider()

    bullet_vecs = np.array([
        get_or_create_embedding("cv_bullet_embeddings", ..., text, provider.name, lambda t: provider.embed([t])[0])
        for text in cv_bullets
    ])
    req_vecs = np.array([
        get_or_create_embedding("job_embeddings", ..., text, provider.name, lambda t: provider.embed([t])[0])
        for text in requirements
    ])

    matrix = bullet_vecs @ req_vecs.T
    # unchanged from Chapter 5 from here down
    ...
```

The `...` in place of an identifier is deliberate: decide it yourself per the exercise's fourth step.

</details>

### Exercise 3 (optional to complete): prove the model-name scoping actually works

Verify, rather than assume, that switching providers can't silently hand back the wrong model's vector.

**What to do:** write a short throwaway script (or a `pytest` test, if you'd rather practice that) that embeds the same CV bullet text twice through your get-or-create path, once with `EMBEDDING_PROVIDER=mock` and once with `EMBEDDING_PROVIDER=huggingface`. Confirm two distinct rows exist in `cv_bullet_embeddings` for that text (one per model name), and confirm that querying with each provider active returns that provider's own vector, not the other one's.

**Pitfalls for this exercise:** remembering that changing `EMBEDDING_PROVIDER` requires restarting whatever process reads `config.py` (or reloading it), since environment variables are read once at import time; comparing vectors by dimension alone isn't enough of a check, since two providers could coincidentally share a dimension while still being semantically different models, so check `model_name` directly instead.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** You already wrote the row-counting verification style in Exercise 1; extend it by also printing or asserting on the `model_name` column for the rows you find, not just the count, so you're proving they're distinct rows for a reason, not just that two rows happen to exist.

**Approach / pseudocode:**
```
embed a fixed piece of text under one provider, via the cache path
switch to the other provider
embed the same text again, via the cache path
look up how many rows exist for that text now, and what model_name each one has
confirm there are two rows, with two different model_name values
```

</details>

## Common pitfalls

1. **Treating the (text, model) pair as (text) alone.** If your lookup query only filters on `content_hash`, switching providers returns stale vectors from the wrong model instead of a fresh embedding. Always filter on both.
2. **No unique index, so nothing actually enforces the invariant your Python code assumes.** Your read-then-write logic can still race or bug its way into duplicates; the unique index is a backstop, not just documentation.
3. **Losing list order when converting cache-hit/cache-miss results back into a matrix.** The matrix math has no way to tell you the order is wrong; it just silently produces a matrix that scores the wrong bullet against the wrong requirement. Verify order explicitly with a known small example before trusting it on real input.
4. **Hashing a Python string directly instead of encoded bytes.** `hashlib.sha256("text")` raises a `TypeError`; you need `.encode("utf-8")` first.

## Further reading

- Python `hashlib` docs: <https://docs.python.org/3/library/hashlib.html>
- PostgreSQL `ON CONFLICT` (upsert): <https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT>
- PostgreSQL multicolumn (compound) indexes: <https://www.postgresql.org/docs/current/indexes-multicolumn.html>
- A short conceptual read on content-addressable storage (the idea behind Git's own object store, if you want the connection to something you already use daily): <https://git-scm.com/book/en/v2/Git-Internals-Git-Objects>

## Checkpoint

Before moving to Chapter 7, you should have:
- [ ] A `content_hash` column and a compound unique index on `(content_hash, model_name)` on `cv_bullet_embeddings` and `job_embeddings`
- [ ] A `db.py` function that looks up a stored vector by `(content_hash, model_name)`, returning nothing found when there isn't one
- [ ] A get-or-create helper that reuses a stored vector on a hit and embeds-then-stores on a miss
- [ ] `/match` using that helper for both CV bullets and JD requirements, with the matrix math otherwise unchanged from Chapter 5
- [ ] Verified that calling `/match` twice with identical input adds zero new rows the second time
- [ ] Verified that the same text under two different providers produces two distinct rows, correctly scoped by `model_name`
- [ ] An understanding of why content-addressing makes cache invalidation free, and why the model name has to be part of the key
- [ ] Code committed to your repo
