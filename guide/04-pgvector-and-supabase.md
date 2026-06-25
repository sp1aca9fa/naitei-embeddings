# Chapter 4: pgvector and Supabase

## What we're building in this chapter

A `src/db.py` module that connects to your Supabase Postgres database, plus the database schema itself: the `pgvector` extension enabled, and three tables (`job_embeddings`, `cv_bullet_embeddings`, `cv_sentence_embeddings`) that store embedding vectors alongside the text they came from. By the end you will be able to take a vector out of `provider.embed(...)`, write it to Postgres, and run a similarity search in SQL that finds the nearest stored vectors to a query.

This is the first chapter where your code talks to a real external system that persists data. Everything up to now lived in memory and vanished when the script ended. From here on, embeddings have a home.

## Why this matters

An embedding you compute and throw away is useless for a real service. The whole point of Feature A (cross-language match) and Feature B (skill canonicalization) is that you embed something once, store it, and compare against it many times later. A job description does not change between the moment you index it and the moment a CV is matched against it, so re-embedding it on every request would be wasteful and slow. You compute the vector once, store it, and reuse it. That "embed-once" pattern is the spine of Chapter 6, and it depends on having a vector store now.

The natural question is "why store vectors in Postgres instead of a dedicated vector database like Pinecone or Weaviate?" For this project the answer is concrete: Naitei already runs on Supabase, which is Postgres. Your jobs, CV versions, and users already live there. If you put embeddings in a separate vector DB you would have two databases to keep in sync, two sets of credentials, two failure modes, and you would lose the ability to write a single SQL query that joins an embedding to the job it belongs to. `pgvector` lets Postgres itself store and search vectors, so embeddings sit right next to the data they describe. At your scale (thousands of jobs and CV bullets, not billions) a dedicated vector DB buys you nothing and costs you complexity. This is the same reasoning a lot of production teams land on.

## Concepts

### What pgvector actually is

`pgvector` is a Postgres extension. An extension is a package of extra functionality you turn on inside a database with `CREATE EXTENSION`. Once enabled, Postgres gains a new column type, `vector(n)`, which stores a fixed-length array of `n` floating-point numbers, and a set of distance operators for comparing two vectors. That is the entire core of it: a column type plus distance math, running inside the database you already have.

A table column declared `vector(768)` holds one 768-dimensional embedding per row. The number must be fixed per column. You cannot put a 768-dim vector in one row and a 1536-dim vector in the next; Postgres rejects it. This is the first place the provider abstraction touches the database: the column dimension must match `provider.dimension`. We will come back to this tension below.

### Distance operators and how they relate to cosine

`pgvector` gives you three operators for comparing vectors in a query:

- `<->` is L2 (Euclidean) distance.
- `<#>` is negative inner product.
- `<=>` is cosine distance.

We use `<=>`, cosine distance, because the whole project is built on cosine similarity. The relationship is simple:

```
cosine_distance = 1 - cosine_similarity
```

So a cosine distance of `0` means identical direction (similarity 1.0), and a distance of `1` means orthogonal (similarity 0). When you want the most similar rows, you order by cosine distance ascending and take the first N:

```sql
SELECT content
FROM job_embeddings
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

That reads as "give me the five rows whose embedding is closest, by cosine distance, to this query vector." If you want the actual similarity score in the output, compute `1 - (embedding <=> query)` as a selected column. Remember the contract from Chapter 3: every vector you store is unit-normalized, which is what makes cosine the right and cheap choice here.

### Indexing: IVFFlat vs HNSW

Without an index, a similarity query compares the query vector against every row in the table. That is a sequential scan, and it is fine for a few hundred rows. As the table grows it gets slow, because the cost grows linearly with the row count. An index makes similarity search approximate but fast: instead of checking every row, it checks a smaller candidate set that is very likely to contain the true nearest neighbors.

`pgvector` offers two index types.

**IVFFlat** divides the vectors into a number of lists (clusters) and, at query time, only searches the lists nearest the query. You must pick the number of lists up front, and crucially you should build the index *after* the table already has a representative amount of data, because the clustering is computed from the existing rows. Build it on an empty table and the clusters are meaningless.

**HNSW** (Hierarchical Navigable Small World) builds a graph that links each vector to its neighbors at multiple levels of granularity, and walks that graph to find nearest neighbors. It builds incrementally as you insert, needs no pre-existing data, gives better recall (it finds the true nearest neighbors more reliably), and is the more forgiving choice. It uses more memory and is slower to build, but at our scale neither matters.

For this project, use **HNSW**. You will be inserting rows incrementally as jobs and CVs come in, you do not have a large training set sitting around, and you would rather not think about tuning a `lists` parameter. The syntax to create one, with cosine as the distance:

```sql
CREATE INDEX ON job_embeddings USING hnsw (embedding vector_cosine_ops);
```

`vector_cosine_ops` tells the index to optimize for the `<=>` operator. There are matching `vector_l2_ops` and `vector_ip_ops` for the other two operators; you want the cosine one because that is what you query with. An index built for one operator does not accelerate the others.

### Schema design: three tables and the parent_bullet_id link

Feature A needs embeddings at two granularities on the CV side and one on the job side:

- **`job_embeddings`** holds embeddings of job-description requirement sentences. One row per requirement sentence.
- **`cv_bullet_embeddings`** holds embeddings of whole CV bullets. One row per bullet.
- **`cv_sentence_embeddings`** holds embeddings of individual sentences split out of multi-sentence bullets. One row per sentence, with a `parent_bullet_id` column pointing back to the bullet it came from in `cv_bullet_embeddings`.

The reason for both bullet-level and sentence-level CV embeddings: a single CV bullet might be "Built a React frontend and migrated the backend to Go." That bullet as a whole has one meaning, but it also contains two distinct claims. Sometimes you want to match a JD requirement against the whole bullet; sometimes against the specific sentence. Storing both, linked by `parent_bullet_id`, lets later queries choose the granularity. The split logic itself (turning one bullet into sentences) is Chapter 5's job. This chapter just builds the tables that will hold the result.

Each table needs, at minimum:

- a primary key,
- the source text (so you can read back what a vector represents without joining elsewhere),
- the `embedding vector(768)` column,
- the model name that produced it (the `name` from your provider, so you know which model an old row used and can re-embed if you switch models),
- a created-at timestamp.

`cv_sentence_embeddings` additionally needs `parent_bullet_id`, a foreign key referencing `cv_bullet_embeddings`.

A design decision worth being deliberate about: these tables live in the same Postgres database as Naitei's own tables (jobs, cv_versions, etc.), because you share the Supabase project. You *could* add hard foreign keys from `job_embeddings` to Naitei's `jobs` table. For this project, do not. Store the Naitei identifier as a plain column (for example `job_id uuid`) without a foreign-key constraint across to Naitei's tables. Reasons: this service does not own those tables, you do not want a schema change here to require coordinating with Naitei, and a stale reference is recoverable while a broken cross-service FK constraint is a deployment headache. Keep the coupling loose on purpose.

### The dimension-must-match-provider tension

Your `vector(n)` columns hard-code a dimension. The mock and HuggingFace providers both produce 768-dim vectors; OpenAI's `text-embedding-3-small` produces 1536. If you declare your columns `vector(768)` and then switch `EMBEDDING_PROVIDER` to `openai`, your inserts will fail, because a 1536-dim vector does not fit a 768-dim column.

For this chapter, declare the columns `vector(768)` to match the mock and e5 defaults you are actually using. Just understand the constraint clearly: the database schema is tied to the embedding model's dimension. Switching models with a different dimension is a schema migration, not just an env-var flip. In a mature system you would handle this by storing the model name per row (which you are) and either keeping a separate column or table per dimension, or re-embedding everything during the migration. We are not solving that now. We are just not pretending it does not exist.

### Python aside: imports, packages, and why Chapter 3 fought you

You mentioned the multi-file structure tripped you up. Here is the model to hold in your head, because this chapter adds another file (`src/db.py`) that imports from your existing ones.

A **module** is a single `.py` file. A **package** is a directory with an `__init__.py` file in it; the `__init__.py` is what makes Python treat the directory as importable, and its contents run when the package is first imported. Your `src/` is a package, and `src/providers/` is a sub-package.

There are two import styles:

- **Absolute**: `from src.providers import get_provider`. Spelled out from the project root. This is what you used in `try_provider.py` and your tests.
- **Relative**: `from .base import EmbeddingProvider`. The leading dot means "from the current package." You used this inside `src/providers/__init__.py`. Relative imports only work *inside* a package, when the file is being run as part of that package, not when run directly as a script.

That last point is the root of the `sys.path.insert(...)` line you wrote at the top of `try_provider.py`. When you run `python scripts/try_provider.py`, Python sets the import root to the `scripts/` directory, so `import src` is not found. Inserting the project root onto `sys.path` fixes it. When you instead run `python -m pytest` from the project root, the root is already on the path, which is why your tests did not strictly need the insert. Same underlying mechanic, different entry points.

**Circular imports** happen when module A imports module B while B imports A. Python starts loading A, hits the import of B, starts loading B, B hits the import of A, finds A only half-loaded, and you get an ImportError or a half-initialized module. The usual fixes, in order of preference: (1) restructure so the dependency only goes one direction, (2) import the *module* rather than a name from it (`from src import config` then `config.database_url`, instead of `from src.config import database_url`), which defers the attribute lookup until call time, or (3) move the import inside the function that needs it so it runs later. For this chapter, `src/db.py` will import `src.config` and that is a one-directional dependency, so you should not hit a cycle. Keep the rule in mind anyway: config and db should not import each other.

### Python aside: context managers and the connection lifecycle

`psycopg2` (the Postgres driver you will use) hands you a **connection** and, from it, **cursors**. A cursor is the object you actually run SQL through and read results from. Both connections and cursors hold real resources (a network socket, a server-side cursor) that must be released. Forgetting to close them leaks connections, and a database has a limited pool of them.

Python's tool for "acquire a resource, use it, guarantee cleanup even on error" is the **context manager**, used with the `with` statement. You have seen it with files (`with open(...) as f:`). It works the same here:

```python
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        print(cur.fetchone())
```

One psycopg2-specific wrinkle to know: using a connection as a context manager (`with conn:`) does **not** close the connection on exit. It commits the transaction if the block succeeds and rolls back if it raises. The connection stays open. The cursor context manager (`with conn.cursor() as cur:`) does close the cursor on exit. So a common, correct shape is to manage the cursor with `with`, commit through the connection, and close the connection explicitly (or hand it out from a helper and let the caller close it). Do not over-think this now; the examples below show a working pattern you can adopt.

## The tools we're using

### `psycopg2` (via `psycopg2-binary`)

- What it is: the most widely used PostgreSQL driver for Python. It speaks the Postgres wire protocol and gives you connections, cursors, and parameterized queries.
- What it does for us: this is how we run raw SQL, including the `pgvector` similarity queries that `supabase-py` cannot express.
- Install: `pip install psycopg2-binary`, then pin in `requirements.txt`. The `-binary` variant ships precompiled so you do not need Postgres dev headers on your machine.
- Docs: <https://www.psycopg.org/docs/>
- Key items we'll use:
  - `psycopg2.connect(dsn)` where `dsn` is your connection string. Returns a connection.
  - `conn.cursor()` returns a cursor.
  - `cur.execute(sql, params)` runs a query. Pass parameters as a tuple, never by string-formatting them in (see the injection note in pitfalls).
  - `cur.fetchone()` / `cur.fetchall()` read result rows.
  - `conn.commit()` persists a transaction.

### `pgvector` (the Python adapter)

- What it is: a small package that teaches `psycopg2` how to convert between Python sequences/NumPy arrays and Postgres's `vector` type.
- What it does for us: after one setup call, you can pass a NumPy array straight into a query parameter and read a `vector` column back as a NumPy array, with no manual string formatting like `'[0.1,0.2]'`.
- Install: `pip install pgvector`, then pin.
- Docs: <https://github.com/pgvector/pgvector-python>
- Key items we'll use:
  - `from pgvector.psycopg2 import register_vector` then `register_vector(conn)`. Call it once per connection, right after connecting. Without it, passing a NumPy array as a parameter will fail or insert garbage.

### `supabase` (supabase-py) — previewed now, used later

- What it is: the official Supabase client for Python. It wraps Supabase's REST and auth APIs.
- What it does for us: in later chapters, when this service needs to read Naitei's own tables (jobs, CV versions) for the backfill in Chapter 9, `supabase-py` is the convenient client for those ordinary row queries. It does not handle `pgvector` similarity search, which is why vector work goes through `psycopg2`.
- Install: `pip install supabase`, then pin. Install it now so it is ready, but you will not write Supabase-client code in this chapter.
- Docs: <https://github.com/supabase/supabase-py>

### The Supabase SQL editor

- What it is: a web-based SQL console in your Supabase project dashboard.
- What it does for us: this is where you will run the one-time `CREATE EXTENSION` and `CREATE TABLE` / `CREATE INDEX` statements. Schema setup is a one-off administrative action, not something your Python code does at runtime, so running it by hand in the editor is the right move.

## How it fits together

```
        provider.embed(["React developer"])   (Chapter 3)
                        |
                        v
              numpy array, shape (1, 768), unit-normalized
                        |
                        v
        +-------------------------------------+
        |             src/db.py               |
        |   get_connection()  -> psycopg2     |
        |   register_vector(conn)             |
        |   INSERT ... VALUES (%s, %s, %s)     |
        +------------------+------------------+
                           |
                           v
        +-------------------------------------+
        |   Supabase Postgres + pgvector      |
        |                                     |
        |   job_embeddings(                   |
        |     id, job_id, content,            |
        |     embedding vector(768),          |
        |     model_name, created_at)         |
        |   + HNSW index on embedding         |
        +------------------+------------------+
                           |
            similarity query: ORDER BY embedding <=> %s LIMIT N
                           |
                           v
              nearest stored rows, with 1 - distance as score
```

The connection string lives in `.env` as `DATABASE_URL` (gitignored), read through your existing `src/config.py` in exactly the same pattern as the provider settings. `src/db.py` reads that config, opens connections, and runs SQL. The schema (extension, tables, index) is created once by hand in the Supabase SQL editor and is not Python's concern at runtime.

## Code examples

These run against a throwaway `items` table, not your real project tables. They show the mechanics of each tool. You will write the real `src/db.py` and the real schema yourself in the exercises.

### Example 1: A toy table and the extension

Run in the Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content text NOT NULL,
    embedding vector(3) NOT NULL
);
```

`CREATE EXTENSION IF NOT EXISTS vector` enables `pgvector`; the `IF NOT EXISTS` makes it safe to run twice. The toy embedding is `vector(3)` so the examples are readable by hand.

### Example 2: Connecting and inserting a vector

```python
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://user:pass@host:5432/postgres")
register_vector(conn)

vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

with conn.cursor() as cur:
    cur.execute(
        "INSERT INTO items (content, embedding) VALUES (%s, %s)",
        ("hello", vec),
    )
conn.commit()
conn.close()
```

Two things to notice. The `%s` placeholders are filled from the tuple passed as the second argument to `execute`; psycopg2 handles quoting and type conversion. Because you called `register_vector(conn)`, the NumPy array is accepted directly for the `vector` column. And `conn.commit()` is what actually persists the insert; without it the transaction rolls back when the connection closes.

### Example 3: A similarity query

```python
query_vec = np.array([0.1, 0.2, 0.25], dtype=np.float32)

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT content, 1 - (embedding <=> %s) AS similarity
        FROM items
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (query_vec, query_vec),
    )
    for content, similarity in cur.fetchall():
        print(f"{similarity:.3f}  {content}")
```

The same `query_vec` is passed twice because it appears twice in the SQL: once to compute the similarity score in the SELECT, once to order by distance. `ORDER BY embedding <=> %s` sorts nearest-first; `1 - (embedding <=> %s)` turns the distance back into a cosine similarity for display.

### Example 4: Reading config the same way you already do

```python
import os
from src import config   # your existing module

# in config.py, alongside your provider settings:
#   database_url = os.getenv("DATABASE_URL")

conn = psycopg2.connect(config.database_url)
```

This mirrors the pattern you built in Chapter 3. The connection string is just another environment variable, read once in `config.py`, used wherever needed. Note the absolute import `from src import config`, which keeps the dependency one-directional and avoids the circular-import trap.

## Your tasks

### Exercise 1: Replication — schema and the db module

The goal of this exercise is to stand up the schema and write a small `src/db.py` that can connect, insert one embedding, and read it back. You are practicing exactly what the chapter taught: the `vector` column type, the connection lifecycle, and `register_vector`.

**1. Get your connection string.** In your Supabase project dashboard, go to Project Settings, then Database, and find the connection string (URI form). Copy it and substitute your database password. Add it to `.env` as `DATABASE_URL=...`. Add a matching empty `DATABASE_URL=` line to `.env.example`. Confirm with `git status` that `.env` is still ignored.

**2. Update `requirements.txt`** with `psycopg2-binary`, `pgvector`, and `supabase`, pinned to their current versions, then install.

**3. Add `database_url` to `src/config.py`**, read from the environment, following the same style as your existing settings. No default; a missing connection string should surface as a clear failure when used, not be papered over.

**4. Enable the extension and create the schema.** In the Supabase SQL editor, run `CREATE EXTENSION IF NOT EXISTS vector`, then create the three tables described in the Concepts section: `job_embeddings`, `cv_bullet_embeddings`, and `cv_sentence_embeddings`. Decide the columns yourself based on the design discussion. Requirements:

- Every table has a primary key, the source text, an `embedding vector(768)` column, a `model_name` text column, and a `created_at timestamptz` defaulting to now.
- `job_embeddings` has a `job_id` column (use `uuid` or `text`; no foreign key to Naitei tables).
- `cv_bullet_embeddings` has whatever identifier ties a bullet to its CV (your call; a `cv_version_id` column is reasonable).
- `cv_sentence_embeddings` has a `parent_bullet_id` that is a real foreign key referencing `cv_bullet_embeddings`.
- Create an HNSW index with `vector_cosine_ops` on the `embedding` column of each table.

**5. Write `src/db.py`** with at least:

- A function that returns a live, vector-registered connection. Think about its signature and name; it should read `config.database_url` and call `register_vector` before handing the connection back.
- A function that inserts one embedding row into `job_embeddings` given the text, the vector, the model name, and a job id. It should use a parameterized query (`%s`), not string formatting.
- A function that fetches a row back by id and returns its content and embedding.

Use the `logging` module for any status messages, not `print` (project style). Type-hint every signature. NumPy arrays are the vector type flowing in and out.

**6. Verify the round-trip.** Write `scripts/try_db.py` that uses `get_provider()` to embed a single string, inserts it into `job_embeddings` via your `src/db.py` functions, reads it back, and prints both the stored content and the norm of the retrieved vector (which should be ~1.0). Run it with `EMBEDDING_PROVIDER=mock` first (fast, no model download), then with `huggingface` if you want to see a real vector make the trip.

How to verify it worked: the script prints the content you inserted and a norm near 1.0, and a `SELECT count(*) FROM job_embeddings;` in the SQL editor shows the row.

### Exercise 2: Application — a working similarity search

This is the payoff. You will store several job-requirement sentences, then search them by meaning. This exercises the `<=>` operator, the ordering, and turning distance into a similarity score, and it is the first time you see cross-language matching come out of the database rather than out of an in-memory array.

**1. In `scripts/try_db.py` (or a new `scripts/try_search.py`),** embed and insert a handful of short job-requirement sentences into `job_embeddings`. Include a mix of English and Japanese, for example a few like "Experience with React", "TypeScript の実務経験", "Backend API design", "チームでの開発経験". Use `get_provider()` so the provider stays swappable.

**2. Write a search function in `src/db.py`** that takes a query string and an integer N, embeds the query with the configured provider, and runs a similarity query against `job_embeddings` returning the top N rows with their similarity scores. Return the results to the caller; do the printing in the script, not in `db.py`.

**3. Drive it from the script.** Query with something like "Frontend developer with React" and print the ranked results with scores. Run it under `huggingface` (the mock will give near-random rankings, by design; only a real model produces meaningful similarity). You should see the React-related and TypeScript-related sentences rank above the unrelated ones, and the Japanese sentences should rank sensibly against the English query. That cross-language ranking, coming straight out of Postgres, is the concrete thing this whole project is built to do.

**4. Reflect (no code).** You stored `model_name` on every row. Why does it matter that a similarity query only compares vectors produced by the *same* model? What goes wrong if you search e5 vectors using an OpenAI query vector? Hold the answer; it is the reason the backfill in Chapter 9 has to be careful.

### Exercise 3 (optional): a reusable, safe insert across tables

Only do this if you want the extra practice; it is not required to proceed. You have three tables with near-identical insert logic. Writing three almost-copy-paste insert functions is a smell. Try writing one insert helper that takes the target table name as an argument.

The catch, and the actual lesson: you **cannot** pass a table or column name as a `%s` parameter. Parameters are for values, not identifiers. Naively f-string-ing the table name into the SQL reopens the injection door you closed by using `%s` for values. The correct tool is `psycopg2.sql`:

```python
from psycopg2 import sql

query = sql.SQL("INSERT INTO {} (content, embedding) VALUES (%s, %s)").format(
    sql.Identifier(table_name)
)
```

`sql.Identifier` safely quotes an identifier; the `%s` values still go through `execute`'s parameter tuple as before. Build the helper, restrict the allowed table names to your three (do not let an arbitrary string through), and rewrite your earlier insert to call it. This is a small but real piece of Python you will reuse.

## Common pitfalls

1. **Forgetting `register_vector(conn)`.** Symptom: errors about adapting NumPy arrays, or vectors stored as a malformed string. Fix: call `register_vector(conn)` once, immediately after `psycopg2.connect`, on every connection. Putting it inside your `get_connection` helper means you never forget.

2. **Forgetting `conn.commit()`.** Symptom: your script prints success, but `SELECT count(*)` shows zero rows. Inserts live in an uncommitted transaction that rolls back when the connection closes. Fix: commit after writes. If you use `with conn:` as a context manager, the successful exit commits for you, which is one reason that pattern is handy.

3. **String-formatting values into SQL.** Symptom: it appears to work, until a value contains a quote, and meanwhile you have an SQL-injection hole. Never do `f"... VALUES ('{content}')"`. Always use `%s` placeholders and pass values as the tuple argument to `execute`. Identifiers (table/column names) are the one exception and need `psycopg2.sql.Identifier`, not `%s`.

4. **Dimension mismatch.** Symptom: `expected 768 dimensions, not 1536` (or vice versa) on insert. The provider's dimension and the column's `vector(n)` disagree. Fix for this chapter: keep columns at `vector(768)` and use the mock or huggingface provider. Do not point it at OpenAI without changing the schema.

5. **Building an HNSW index but querying with the wrong operator.** Symptom: queries are slow even with an index. The index is built `vector_cosine_ops` but you queried with `<->` (L2), so the index does not apply. Fix: query with `<=>` to match the cosine index.

6. **Connection string confusion (pooler vs direct).** Supabase shows more than one connection string. The direct connection and the session pooler both work for these scripts. If you later deploy a long-running service, you will revisit this. For now, use what the dashboard labels as the connection string and make sure the password is filled in.

7. **Committing `.env` with the database password.** Symptom: your DB credentials land on GitHub. The connection string contains your password. `.env` is already gitignored from Chapter 0; verify with `git status` that only `.env.example` (with an empty `DATABASE_URL=`) is ever staged.

8. **`psycopg2` install fails building from source.** Symptom: compiler errors mentioning `pg_config`. Fix: install `psycopg2-binary`, not `psycopg2`. The binary wheel needs no local Postgres headers.

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge</summary>

Split the work in two halves that do not depend on each other in code: the schema (SQL you run by hand in the Supabase editor, once) and the Python (`src/db.py`, which assumes the schema already exists). Get the schema right and verified first by inserting a row by hand in the SQL editor, then write Python against tables you know are correct. Debugging "is it my SQL or my Python" is much easier when you have already proven the SQL half independently.

For `src/db.py`, you only need a handful of small functions. A connection helper, an insert, a fetch-by-id, and (Exercise 2) a search. None of them is long. If a function is growing past fifteen lines, you are probably mixing concerns; split it.

</details>

<details>
<summary>Hint 2 — Schema sketch</summary>

This is the shape, not a copy-paste solution. Decide names and exact types yourself.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE job_embeddings (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id uuid,
    content text NOT NULL,
    embedding vector(768) NOT NULL,
    model_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON job_embeddings USING hnsw (embedding vector_cosine_ops);

-- cv_bullet_embeddings: like above, with a cv_version_id instead of job_id.

-- cv_sentence_embeddings: like cv_bullet_embeddings, plus
--   parent_bullet_id bigint REFERENCES cv_bullet_embeddings(id)
-- and its own HNSW index.
```

</details>

<details>
<summary>Hint 3 — db.py structure</summary>

```python
import logging

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from src import config

logger = logging.getLogger(__name__)


def get_connection() -> psycopg2.extensions.connection:
    """Open a Postgres connection with pgvector type support registered."""
    conn = psycopg2.connect(config.database_url)
    register_vector(conn)
    return conn


def insert_job_embedding(
    job_id: str, content: str, embedding: np.ndarray, model_name: str
) -> None:
    """Store one job-requirement embedding."""
    # open connection, execute parameterized INSERT, commit, close
    ...


def get_job_embedding(row_id: int) -> tuple[str, np.ndarray]:
    """Fetch the content and embedding for one row by id."""
    ...
```

Leave the bodies for yourself. The insert body is one `execute` with a four-value tuple and a commit. The fetch is one `execute` with a one-value tuple and a `fetchone`.

</details>

<details>
<summary>Hint 4 — the search query</summary>

```python
def search_jobs(query: str, n: int) -> list[tuple[str, float]]:
    """Return the top-n job rows most similar to query, as (content, score)."""
    from src.providers import get_provider

    provider = get_provider()
    query_vec = provider.embed([query])[0]   # shape (768,)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, 1 - (embedding <=> %s) AS similarity
                FROM job_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, query_vec, n),
            )
            return cur.fetchall()
    finally:
        conn.close()
```

Note `provider.embed([query])[0]`: `embed` is always batch-shaped (Chapter 3), so you pass a one-element list and take row zero to get a 1-D vector for the query. The local import of `get_provider` inside the function is one way to keep `db.py`'s top-level imports free of the providers package; an absolute import at the top would also be fine since there is no cycle.

</details>

## Further reading

- pgvector README, especially the indexing and querying sections: <https://github.com/pgvector/pgvector>
- pgvector-python adapter docs: <https://github.com/pgvector/pgvector-python>
- Supabase's own pgvector guide: <https://supabase.com/docs/guides/database/extensions/pgvector>
- psycopg2 on passing parameters safely: <https://www.psycopg.org/docs/usage.html#passing-parameters-to-sql-queries>
- A readable explanation of HNSW vs IVFFlat trade-offs: the "Indexing" section of the pgvector README above covers it concisely.

## Checkpoint

Before moving to Chapter 5, you should have:

- [ ] The `vector` extension enabled in your Supabase project
- [ ] Three tables created (`job_embeddings`, `cv_bullet_embeddings`, `cv_sentence_embeddings`) with `vector(768)` columns, `model_name`, timestamps, the `parent_bullet_id` foreign key on the sentence table, and an HNSW `vector_cosine_ops` index on each embedding column
- [ ] `DATABASE_URL` in `.env` (and an empty placeholder in `.env.example`), with `.env` confirmed gitignored
- [ ] `database_url` read in `src/config.py`
- [ ] `src/db.py` with a vector-registered connection helper, an insert function, a fetch-by-id function, and a similarity search function, all parameterized and type-hinted
- [ ] `scripts/try_db.py` proving a full round-trip: embed, insert, read back, norm ~1.0
- [ ] A working similarity search that ranks related and cross-language sentences sensibly under the huggingface provider
- [ ] An understanding of why the column dimension is coupled to the provider, and why `model_name` is stored per row
- [ ] Code committed to your repo
