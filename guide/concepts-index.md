# Concepts Index

Running ledger of every concept/tool introduced, and which chapter taught it. Used to pick genuinely non-adjacent material for each chapter's "Quick recall check" and Exercise 4 retrieval callback.

Update this after finalizing every chapter: append one line per new concept/tool.

## Chapter 0: Project Setup
- Virtual environments (`venv`) — Chapter 0
- ASGI and why FastAPI exists — Chapter 0
- `uvicorn` (ASGI server) — Chapter 0
- The 12-factor `.env` pattern — Chapter 0
- `pip` and `requirements.txt` — Chapter 0
- Minimal FastAPI app and routes — Chapter 0

## Chapter 1: Embeddings — Concept and First Use
- What an embedding is — Chapter 1
- Vector spaces — Chapter 1
- Cosine similarity — Chapter 1
- Why not Euclidean distance — Chapter 1
- `sentence-transformers` basics — Chapter 1
- `numpy` basics (vectors) — Chapter 1
- The `all-MiniLM-L6-v2` model — Chapter 1

## Chapter 2: Multilingual Embeddings
- Why most embedding models are English-only — Chapter 2
- What multilingual models do differently — Chapter 2
- The `intfloat/multilingual-e5-base` model — Chapter 2
- The `query:` / `passage:` prefix quirk — Chapter 2
- Similarity score ranges differ by model — Chapter 2
- Minimal nearest-neighbor lookup — Chapter 2

## Chapter 3: Provider Abstraction
- Provider abstraction pattern (why abstract over models) — Chapter 3
- Python abstract base classes (`abc`) — Chapter 3
- Deterministic mocks: hashing text to seed a random generator — Chapter 3
- Environment-driven configuration (`python-dotenv`) — Chapter 3
- `openai` Python SDK basics — Chapter 3
- `hashlib` (first introduction) — Chapter 3
- Where the e5 prefix should live in the provider layer — Chapter 3
- Async vs sync (why not using async yet) — Chapter 3

## Chapter 4: pgvector and Supabase
- What pgvector is — Chapter 4
- Distance operators (`<=>`) and how they relate to cosine — Chapter 4
- Indexing: IVFFlat vs HNSW — Chapter 4
- Schema design: three tables, `parent_bullet_id` link — Chapter 4
- Dimension-must-match-provider tension — Chapter 4
- Imports, packages, circular imports (Python aside) — Chapter 4
- Context managers and connection lifecycle (Python aside) — Chapter 4
- `psycopg2` / `psycopg2-binary` — Chapter 4
- `pgvector` Python adapter (`register_vector`) — Chapter 4
- `supabase-py` (previewed) — Chapter 4
- `psycopg2.sql.Identifier` for dynamic table/column names — Chapter 4

## Chapter 5: Building the FastAPI Service
- The match problem as a matrix — Chapter 5
- Cosine similarity as a dot product (batched via `@`) — Chapter 5
- Sentence splitting and why not `.split(".")` — Chapter 5
- Pydantic models as the API contract — Chapter 5
- `def` vs `async def` in FastAPI — Chapter 5
- `pysbd` (sentence segmentation) — Chapter 5
- Per-column indexing into a similarity matrix, `np.argsort` for ranking — Chapter 5
- `pydantic.Field` with `ge`/`le` validation — Chapter 5

## Chapter 6: Connecting Match to Storage
- Content-addressed caching (hash as cache key) — Chapter 6
- Compound cache keys: why model name must be part of the key — Chapter 6
- Schema evolution: `ALTER TABLE`, compound unique index — Chapter 6
- Read-then-write races and `ON CONFLICT` — Chapter 6
- `hashlib.sha256(...).hexdigest()` for cache keys — Chapter 6
- Stacking a list of 1D vectors into a 2D array (`np.array`, `np.vstack`) — Chapter 6
