# Naitei Embeddings

A Python FastAPI service that adds embedding-based machine learning to [Naitei](https://github.com/), a bilingual job-hunting tool for the Tokyo tech market. It does cross-language semantic matching between English CVs and Japanese job descriptions, plus multilingual skill canonicalization.

This repository is two things at once:

- **A real service.** It is built to be deployed and called by Naitei's TypeScript backend over HTTP.
- **A learning project.** It is also a structured, hands-on way for me to learn Python and applied ML by building something real rather than following disconnected tutorials.

---

## The project: what it does

Naitei is pivoting to serve non-native Japanese speakers who can interview in Japanese but want to job-hunt in English. That creates a hard problem: matching an English CV against a Japanese job description means comparing text across languages, where keyword matching fails. Embeddings solve this by mapping text into a shared vector space where "React developer" and "Reactエンジニア" land close together regardless of language.

Planned capabilities:

- **Cross-language semantic match** (`POST /match`): given a CV (English bullets) and a job description (Japanese or English), return a similarity matrix of CV bullets against JD requirement sentences, the requirements no bullet covers well (coverage gaps), and the top matches per requirement.
- **Skill canonicalization** (`POST /canonicalize-skills`): given skill mentions in any language ("React", "リアクト", "TS", "型付きJS"), return canonical forms via nearest-neighbor lookup, so skill tracking deduplicates across languages.

### How it works

- **FastAPI + uvicorn** serve the HTTP API.
- **A provider abstraction** generates embeddings and is swappable via an env var, mirroring Naitei's own provider-agnostic AI layer:
  - `huggingface` runs `intfloat/multilingual-e5-base` locally (free, handles Japanese and English well).
  - `openai` uses `text-embedding-3-small` (paid, cheap).
  - `mock` returns deterministic fake vectors for fast, offline development and tests.
- **pgvector on Supabase** stores the vectors. The service shares Naitei's existing Supabase Postgres project. Similarity search runs as raw SQL via `psycopg2`, using an HNSW index with cosine distance. Embeddings are computed once and reused.

### Status

Built so far: the FastAPI app with a `/health` endpoint, the three-provider embedding abstraction, and the pgvector storage layer (connect, insert, fetch, and similarity search across the `job_embeddings`, `cv_bullet_embeddings`, and `cv_sentence_embeddings` tables).

Next up: the `/match` and `/canonicalize-skills` endpoints, shared-secret auth, a one-time backfill of existing Naitei data, and deployment.

---

## The learning: how it is being built

I came to this from CS50P, so solid Python basics but no experience with the wider ecosystem (packaging, FastAPI, NumPy, embeddings, pgvector). Instead of copying finished code, the project is worked through as a chapter-by-chapter guide in [`guide/`](guide/). Each chapter teaches one topic, then I write all the real code myself in `src/`, `scripts/`, and `tests/`, commit, and move on.

The guides live as markdown and lean on collapsible hint blocks so I can try each exercise before peeking.

> Tip for reading the guides in VS Code: open the rendered **Markdown preview** with `Ctrl+Shift+V` (or the preview icon at the top right). In preview the "Stuck? Hints" blocks stay collapsed until you click them; in the raw source everything is visible.

Chapters so far:

- `00-setup.md` - venv, structure, the empty FastAPI app
- `01-embeddings-concept.md` - what embeddings are, cosine similarity
- `02-multilingual-embeddings.md` - multilingual models, the e5 query/passage prefixes
- `03-provider-abstraction.md` - the swappable provider design
- `04-pgvector-and-supabase.md` - schema, pgvector, storing and searching vectors

---

## Local development

Run these from the project root.

Create and activate a virtual environment:
```
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Create a `.env` from `.env.example` and fill in your own values (Supabase `DATABASE_URL`, provider selection, etc.). `.env` is gitignored.

Run the server:
```
uvicorn src.main:app --reload
```

## Endpoints

Health check:
```
curl http://localhost:8000/health
```
Returns `{"status": "ok"}`. The `/match` and `/canonicalize-skills` endpoints are added in later chapters.

## Tech stack

Python 3.12, FastAPI, uvicorn, sentence-transformers, OpenAI SDK, Supabase (Postgres + pgvector), psycopg2, pytest.
