# naitei-embeddings — Project Memory for Claude Code

This is the project context file. Read this entire file before doing anything. Re-read it whenever the user reminds you to.

---

## TL;DR

**What:** A Python FastAPI service that adds embedding-based ML capabilities to Naitei (a bilingual job-hunting tool for Japan). The service does cross-language semantic matching between English CVs and Japanese job descriptions, plus multilingual skill canonicalization. Deployed long-term, called by Naitei's TypeScript backend via shared-secret HTTP.

**For whom:** Raphael (the user). Single user. He just finished CS50P and has only basic Python knowledge. He is learning Python and ML through this project. He will write all the code himself.

**Your role:** You are generating a chapter-by-chapter learning guide as markdown files. The user reads each chapter, writes the code himself in `/src`, commits, then asks you for the next chapter. Do NOT generate code for him in the chapter files except as illustrative examples or exercise scaffolding. Do NOT generate more than one chapter at a time. Do NOT generate ahead.

**Timeline:** 2-3 weeks of evening work. Roughly 25-30 hours of focused learning. Do not rush. Depth over speed.

---

## Context: Naitei

Naitei is the user's flagship project: an AI-powered job-hunting dashboard built for the Tokyo tech market. Stack: React + TypeScript (frontend), Node.js + Express + TypeScript (backend), Supabase (PostgreSQL + Auth), Vercel (deployment). It uses a provider-agnostic AI layer supporting Claude, OpenAI, Gemini, Ollama.

The user pastes a job URL or description, Naitei scrapes it, sends the description + the user's resume to Gemini, and Gemini returns a structured fit score (0-100 across 5 categories), an ATS score, matched skills, missing skills, green flags, red flags, and more. Naitei also generates cover letters, interview prep, and tracks applications on a Kanban board.

**Naitei is pivoting** (internal direction, not yet public) to be specifically positioned as a bilingual job-hunting tool for non-native Japanese speakers: people who can interview in Japanese but want to job-hunt in English. This positioning informs the Python service's feature set.

Naitei's repo (separate from this one) lives elsewhere; the user has full access and control. The Python service will share Naitei's Supabase project for data.

---

## The Python Service — What It Builds

### Core features (must build)

1. **FastAPI service with health endpoint.** Standard `/health` returning 200.

2. **Provider abstraction for embedding generation.** Mirrors Naitei's AI provider pattern. Three providers, swappable via env var:
   - **HuggingFace local** — uses `sentence-transformers` library with `intfloat/multilingual-e5-base` model. Free, runs locally, multilingual (handles English + Japanese well).
   - **OpenAI cloud** — uses `text-embedding-3-small`. Paid (very cheap, ~$0.02 per 1M tokens). Multilingual.
   - **Mock** — returns deterministic random-but-reproducible vectors. For development and tests. No model download, no API cost.

3. **Supabase + pgvector storage.** Tables for storing embeddings linked to existing Naitei data. The Python service shares Naitei's existing Supabase project.

4. **Feature A: Cross-language semantic match endpoint.** `POST /match` accepts a CV (English bullets) and a JD (Japanese or English text), returns:
   - A similarity matrix (each CV bullet × each JD requirement sentence with cosine similarity scores)
   - Identified coverage gaps (JD requirements that no CV bullet matches above a threshold)
   - Top-N best matches per JD requirement
   
   Both bullet-level and sentence-level embeddings are computed and stored. Multi-sentence bullets are split into sentences with a `parent_bullet_id` link so both granularities are queryable.

5. **Feature B: Skill canonicalization endpoint.** `POST /canonicalize-skills` accepts a list of skill mentions in any language ("React", "Reactを使用", "リアクト", "TypeScript", "TS", "型付きJS"), returns canonical forms by nearest-neighbor lookup against a canonical skill list. Enables Naitei's skill tracking to deduplicate across languages.

6. **Backfill script.** One-time CLI script to embed all of the user's existing Naitei jobs and CV versions. Runs once after the service is live. Teaches batch operations.

7. **Deployment.** Long-running service on Render or Fly.io (chapter recommends Fly.io; free tier stays warm, no cold-start delay). Proper auth via shared secret. HTTPS-only.

### Optional / extra chapters

These are NOT part of the core path. Generate them only if the user asks. They go at the end of the guide.

- **Extra Chapter 1 (highest priority, basically mandatory): TypeScript changes in Naitei.** How to wire Naitei's backend to call the Python service, store results, and surface them in the React frontend. The user will likely do this immediately after finishing the Python core.

- **Extra Chapter 2: Feature D — 志望動機 coverage scoring.** Endpoint that verifies a generated 志望動機 actually addresses the JD's requirements semantically. Same embedding infrastructure, new endpoint.

- **Extra Chapter 3: Sentry integration.** Production error tracking. Core chapters cover good stdout logging; Sentry is the polish.

### Explicitly OUT of scope

- Web UI on the Python service (it's an API only)
- Multi-user authentication on the Python service (single-user, trust the shared secret)
- External job scraping (data acquisition for Japan jobs is not realistic for free; the user inputs jobs into Naitei manually)
- Feature C (Japanese language difficulty scoring) — interesting but not in this project
- Caching layer (queries are fast enough without it)
- Rate limiting (Naitei is the only client)
- Tests for LLM-side prompts (those stay in Naitei TypeScript)

---

## Technical Decisions (locked in)

- **Python version:** 3.12
- **Web framework:** FastAPI (with Pydantic v2 models for request/response validation)
- **Embedding library:** `sentence-transformers` for local; `openai` Python SDK for cloud
- **Default local model:** `intfloat/multilingual-e5-base` (~280MB download, handles JA+EN well)
- **Vector storage:** pgvector extension in the existing Naitei Supabase project
- **Database access:** `supabase-py` for general queries; raw SQL via `psycopg2` for pgvector similarity queries
- **HTTP client (for OpenAI calls and any outbound HTTP):** `httpx` (not `requests`)
- **File paths:** `pathlib` (not `os.path`)
- **Logging:** standard `logging` module, structured-ish messages to stdout, viewable in Fly.io/Render dashboards
- **Auth between Naitei and Python service:** shared secret in `Authorization: Bearer <secret>` header. Secret generated with `openssl rand -hex 32`, stored in `.env`, gitignored.
- **Code style:** type hints on all function signatures, docstrings on all functions, f-strings, snake_case, 4-space indent, Black-compatible formatting
- **Repo name:** `naitei-embeddings`
- **Branch:** `main`

---

## Repo Structure

```
naitei-embeddings/
├── PROJECT.md              (this file)
├── README.md               (public-facing repo intro)
├── .gitignore
├── .env.example            (template, real .env is gitignored)
├── requirements.txt
├── pyproject.toml          (optional, for tooling config)
├── guide/
│   ├── 00-setup.md
│   ├── 01-embeddings-concept.md
│   ├── 02-...
│   └── (more chapters as they are generated)
├── src/
│   ├── __init__.py
│   ├── main.py             (FastAPI app)
│   ├── config.py           (env var loading)
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py         (abstract EmbeddingProvider)
│   │   ├── huggingface.py
│   │   ├── openai.py
│   │   └── mock.py
│   ├── db.py               (Supabase + pgvector access)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── match.py
│   │   └── canonicalize.py
│   └── (other modules as chapters introduce them)
├── tests/
│   ├── __init__.py
│   ├── test_providers.py
│   ├── test_match.py
│   └── ...
└── scripts/
    └── backfill.py         (one-time data backfill)
```

You may evolve this as chapters introduce concepts. Do not pre-create files the user hasn't been taught to build yet.

---

## Chapter Plan

This is the ordered chapter list. Each chapter teaches one focused topic and produces one focused piece of the project. Do NOT deviate without the user's explicit approval.

### Core path

- **Chapter 0: Project Setup.** Python venv (3.12), folder structure, `requirements.txt` with starter packages, `.gitignore`, `.env.example`, the empty FastAPI app with a `/health` endpoint, the first commit. The user should end this chapter with `uvicorn` serving "OK" from `/health` locally.

- **Chapter 1: Embeddings — Concept and First Use.** What an embedding is, why it exists, what a vector space is, what cosine similarity measures and why we use it (not Euclidean). The `sentence-transformers` library. Loading a model, embedding a single string. Comparing two strings.

- **Chapter 2: Multilingual Embeddings.** Why most embedding models are English-only and what happens if you embed Japanese in one. The `multilingual-e5-base` model specifically. Demonstration: embedding "React developer" in English and "Reactエンジニア" in Japanese and showing they cluster together. The E5 model's quirk of requiring "query:" / "passage:" prefixes — explain why and when to use each.

- **Chapter 3: Provider Abstraction.** Why abstract over providers (matches Naitei's pattern, swap models cheaply, test with mock). Designing the `EmbeddingProvider` abstract base class. Implementing HuggingFace, OpenAI, and Mock providers. Configuration via env vars. The Mock provider's design: deterministic hashing of input text to seed a numpy random generator, so the same text always returns the same fake vector. Critical for tests.

- **Chapter 4: pgvector and Supabase.** What pgvector is, why it lives in PostgreSQL rather than a separate DB. Enabling the extension in Supabase. Designing the schema: `job_embeddings`, `cv_bullet_embeddings`, `cv_sentence_embeddings` tables (with `parent_bullet_id` for sentence-level rows). Vector column types and dimensions. Indexing strategies (IVFFlat vs HNSW — recommend HNSW for this scale). The `supabase-py` Python client basics. Writing a first vector with raw SQL via `psycopg2`.

- **Chapter 5: Building the FastAPI Service.** FastAPI fundamentals: routes, Pydantic models for request/response, automatic OpenAPI docs at `/docs`. Designing the `/match` request schema (CV bullets array, JD text). Designing the response schema (similarity matrix, coverage gaps). Sentence splitting for multi-sentence bullets — covers the `nltk` or `pysbd` library choice; recommend `pysbd` for multilingual support. Building the match logic end to end (without DB yet — in-memory for now).

- **Chapter 6: Connecting Match to Storage.** Adding persistence to `/match`. Storing embeddings on first computation, retrieving on subsequent calls. The "embed-once" pattern: jobs and CV bullets are embedded the first time they're seen, then reused forever. Cache key design (hash of normalized text). When to re-embed (CV bullet edited, JD updated).

- **Chapter 7: Skill Canonicalization.** Building the canonical skill list (a JSON file in the repo, ~200-500 common dev skills as a starting list). The `/canonicalize-skills` endpoint. Implementation: embed all canonical skills once at startup, embed input skills on request, return nearest canonical match by cosine similarity. Threshold tuning (what similarity score counts as "same skill"?). Handling unknown skills (return null + log for later review).

- **Chapter 8: Authentication and Naitei Integration Contract.** Implementing the shared-secret middleware in FastAPI. The integration contract for Naitei: endpoints, request/response shapes, error handling. This chapter defines what Naitei needs to send and what it gets back. Naitei's TypeScript changes are NOT covered here (that's Extra Chapter 1) — this chapter just specifies the contract.

- **Chapter 9: Backfill Script.** Writing `scripts/backfill.py` to embed all existing Naitei jobs and CV versions. Batch processing patterns. Progress logging. Idempotency (re-runnable safely). This is also where the user practices reading Naitei's existing Supabase tables from Python.

- **Chapter 10: Testing.** Unit tests for providers (using the Mock provider). Unit tests for the similarity math. Integration tests for the full `/match` flow against a known CV+JD pair with hand-curated expected matches. Smoke tests for endpoints. Running tests in CI is out of scope (no GitHub Actions setup); just local `pytest`.

- **Chapter 11: Deployment.** Deploying to Fly.io (preferred for free-tier warmness) or Render (alternative). Dockerfile setup. Environment variable management in the platform. Setting up the shared secret. Verifying the service is live and Naitei can reach it. Logging visibility.

### Optional chapters (generate only on request)

- **Extra Chapter 1: TypeScript Integration in Naitei.** How to call the Python service from Naitei's Node backend. New TypeScript service module. Storing match results in Naitei's existing tables. Surfacing similarity matrix and coverage gaps in Naitei's React frontend (job detail page). Error handling when Python service is down.

- **Extra Chapter 2: Feature D — 志望動機 Coverage Scoring.** New endpoint `POST /coverage` that takes a generated 志望動機 + JD and returns coverage analysis. Same embedding infrastructure, new logic.

- **Extra Chapter 3: Sentry Integration.** Adding Sentry for error tracking. Configuration. What to capture and what to ignore.

---

## Chapter Format (mandatory structure)

Every core chapter MUST follow this structure. Deviation requires the user's explicit approval.

```markdown
# Chapter N: [Title]

## What we're building in this chapter

[1-2 sentences: what concrete thing the user will have at the end.]

## Why this matters

[Theoretical / contextual: why is this concept important, where does it fit in the larger project, what real-world systems use it. Aim for the user to understand the WHY before the HOW.]

## Concepts

### [Concept 1]

[Explain the concept. If it has a mathematical or technical foundation, explain enough of it that the user understands the intuition. Use analogies where helpful. Do not assume prior knowledge beyond CS50P-level Python.]

### [Concept 2]

[As above.]

## The tools we're using

### [Package or library 1]

- What it is: [one sentence]
- What it does for us: [one sentence specific to this project]
- Install: `pip install ...`
- Docs: [link]
- Key methods/classes we'll use:
  - `method_or_class_name(...)` — [what it does, with a small code example]
  - ...

### [Package or library 2]

[As above.]

## How it fits together

[Walk through what the code in this chapter does at a high level. Diagrams in ASCII or markdown are welcome. Show the data flow.]

## Code examples

[2-4 small, focused code examples illustrating the concepts. These are NOT the user's solution. They demonstrate what the tools do. Each example should have a short "input → output" demonstration.

Minimize handed-out code, even for new concepts. Keep these examples at the concept level on throwaway/toy structures (a fake `items` table, an `Animal` class, etc.) rather than on the chapter's actual project files, so the user still has to write the real code himself in the exercises. Push fuller, closer-to-the-solution skeletons down into the collapsible "Stuck? Hints" section, where they serve as a safety net the user opts into, rather than the first thing he reads.]

## Your tasks

Every core chapter has at least two hands-on coding exercises, and both Exercise 1 and Exercise 2 must be the user writing code that directly applies what the chapter just taught. Testing is NOT the default second exercise; it is an occasional, optional Exercise 3 (see below). The user writes the code himself in `/src`, `/scripts`, or wherever the chapter directs. Throwaway exploration scripts under `/scripts` are kept as a record of the learning process.

### Exercise 1: Replication

[A first exercise where the user writes code similar to what the chapter just taught. Spell out:
- What files to create or modify
- What functions/classes to write
- What behavior is expected
- How to verify it works (a one-line test, or a curl command, or similar)]

### Exercise 2: Application

[A second exercise where the user must use the chapter's concept to produce an intermediate result and then use that result to reach a final answer. This is still hands-on coding of the chapter's concept, not a test-writing exercise. May be unrelated to the FastAPI service being built. Same level of detail as Exercise 1.]

### Exercise 3 (optional, occasional): Tests or extension

[Include only when it fits naturally and hermetically. Either a test-writing exercise (when the chapter produced something testable without external dependencies) or a small coding extension/refactor that deepens the chapter's concept. Mark it clearly as optional and not required to proceed. Skip it entirely for chapters where it would be forced (for example, a layer that can only be tested against a live external service, which the dedicated testing chapter covers later).]

## Common pitfalls

[3-5 things that commonly go wrong with this topic, and how to fix them. Be specific.]

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge</summary>

[A gentle prompt: "Think about how you'd structure X so Y is true." NO code. Just direction.]

</details>

<details>
<summary>Hint 2 — Approach and pseudocode</summary>

[Higher detail: pseudocode or a structural outline. Tells the user the shape of the solution without writing it.]

</details>

<details>
<summary>Hint 3 — Code skeleton</summary>

[The most help: actual code with one or two key parts blanked out, OR a complete solution if Hint 2 wasn't enough. This is the safety net.]

</details>

## Further reading

[2-4 links: official docs, relevant blog posts, deeper explorations. Optional reading, not required to proceed.]

## Checkpoint

Before moving to Chapter N+1, you should have:
- [ ] [Concrete deliverable from Exercise 1]
- [ ] [Concrete deliverable from Exercise 2]
- [ ] [Any conceptual takeaways the user should have internalized]
- [ ] Code committed to your repo
```

This format is non-negotiable. Every core chapter follows it.

---

## Chapter Generation Workflow (CRITICAL)

The user will work through chapters one at a time. You will generate one chapter at a time. This is the entire workflow:

1. **First-time invocation:** The user asks you to generate Chapter 0. You read this PROJECT.md fully, then generate `guide/00-setup.md` following the chapter format above. You do NOT generate Chapter 1 or anything else.

2. **Subsequent chapters:** The user works through the chapter, writes code in `/src` (and other folders as appropriate), and commits. When ready for the next chapter, they ask you to generate it.

3. **Before generating any chapter N+1**, you MUST:
   - Re-read PROJECT.md
   - Read the existing files in `/src`, `/scripts`, `/tests`, `/guide` to see what the user has actually built
   - Verify that the user's code matches what Chapter N taught (functionally correct, follows the architectural patterns, no major drift)
   - If the user's code has FUNCTIONAL BUGS or MAJOR ARCHITECTURAL DRIFT from what Chapter N described, STOP and report this to the user. Tell them specifically what's wrong and what to fix before proceeding. Do NOT generate the next chapter until the user confirms they've fixed it (or explicitly tells you to proceed anyway).
   - If the user's code differs stylistically (different variable names, different structure within the same approach), that's fine. Adapt the next chapter to use their names and structure.
   - If the user made improvements beyond what Chapter N taught, acknowledge them positively but don't comment extensively.

4. **Never generate more than one chapter at a time.** Even if the user asks for "all chapters" or "the next two chapters", politely refuse and explain that the workflow is one at a time, because each chapter depends on the previous chapter's actual code.

5. **Never get ahead.** If a chapter introduces concept X and a later chapter introduces concept Y that builds on X, don't sneak Y into the X chapter. Each chapter has one focus.

6. **Track progress at the end of each chapter.** End each chapter with the Checkpoint section listing what the user should have. This is also useful for verification in the next chapter.

---

## Code Style Rules

When you DO write code (in examples and hint scaffolding), follow these:

- Python 3.12 syntax
- Type hints on every function signature: `def embed(text: str) -> list[float]:`
- Docstrings on every function and class. Triple-quoted, brief, in English. Describe what it does, not how. Document parameters and return types if non-obvious.
- f-strings for string formatting (`f"Got {count} items"`)
- snake_case for variables and functions
- PascalCase for classes
- 4-space indentation
- `pathlib.Path` for file paths, not `os.path`
- `httpx` for HTTP requests, not `requests`
- `pydantic` v2 for data models — use the `pydantic.BaseModel` class
- Imports at the top, grouped: stdlib, third-party, local
- No `print()` in production code — use the `logging` module

---

## User's Background and Calibration

- **Python knowledge:** basic. Finished CS50P. Knows variables, functions, conditionals, loops, exceptions, file I/O, regex, basic OOP, libraries, unit testing with pytest, basic Flask. Has written one Flask API (his CS50P final project).
- **What he does NOT know well yet:** type hints in depth, async/await, Pydantic, FastAPI specifically, NumPy, the transformers library, pgvector, the Supabase Python client.
- **Other languages:** strong in JavaScript/TypeScript, Ruby on Rails. So concepts like web frameworks, REST APIs, ORMs, deployment are familiar — the Python-specific syntax and idioms are the new part.
- **Communication preferences:** concise, no filler phrases. No em dashes. No emoji. Direct tone. He's a working adult, not a beginner who needs hand-holding on basics, but he genuinely doesn't know Python ecosystem specifics.
- **Language for the guide:** English.

When explaining things:
- Don't over-explain basics he already knows (loops, functions, OOP fundamentals).
- DO explain Python-specific idioms when they appear (decorators, context managers, async, list/dict comprehensions if used in non-obvious ways).
- DO add short, focused asides on non-basic Python ecosystem mechanics that are not the chapter's topic but that he will hit anyway: imports (absolute vs relative), packages and `__init__.py`, circular imports, `sys.path` and script-vs-module execution, virtualenvs, etc. He got stuck on multi-file structure and imports in Chapter 3 (his first Python app spanning more than two files). Flag the likely gotcha proactively rather than waiting for him to trip on it. Keep these asides tight; do not turn the guide into a Python textbook.
- DO explain every new library and every new concept the first time it appears.
- DO connect ideas to his existing knowledge: "this is like Express middleware but for FastAPI" lands well; "this is like a Rails service object" lands well.

---

## How the User Should Ask Claude (the chat assistant) for Help

This section is FOR THE USER, not for Claude Code. Claude Code: if the user mentions getting stuck or struggling, you can point them back to this section.

While working through this project, the user may run into things Claude Code can't easily help with: broader conceptual questions, debugging tricky local issues, deciding whether to deviate from the guide, getting unstuck without using the in-chapter hints, etc. For these, opening a chat with Claude (the assistant — me, the one writing this memory) is the right move.

When asking Claude for help during this project, the user should:

1. **Mention this is the naitei-embeddings project.** Claude has memory of the design conversation but not of the in-progress code. Saying "I'm working on the naitei-embeddings Python ML project" reactivates context.

2. **Share which chapter you're on.** "I'm partway through Chapter 4" gives Claude the right level of context for what concepts you've covered.

3. **Share the actual error or symptom.** Full error messages, not paraphrases. Code snippets, not descriptions.

4. **Share what you've already tried.** Saves a round-trip of Claude suggesting things you've ruled out.

5. **Don't paste the entire PROJECT.md back.** It bloats the context and slows responses. Just reference it: "per the project memory, we decided X, but I'm wondering if Y."

6. **For "should I deviate" questions, share why you're considering it.** "The guide says use HNSW but I read pgvector docs and IVFFlat seems simpler — does it actually matter for our scale?" gets a better answer than "should I use IVFFlat or HNSW?"

7. **For conceptual questions about ML/embeddings, ask freely.** Claude will explain at the right depth without you needing to set up a lot of context. "Why does cosine similarity work better than Euclidean for embeddings?" is a fine standalone question.

---

## Notes for Claude Code on Things to Avoid

- Do not invent features. If a feature isn't in the chapter plan or explicitly requested by the user, it's out.
- Do not rewrite the user's code. Suggest changes in prose; let the user implement them.
- Do not generate the guide in HTML, interactive widgets, or anything other than plain markdown.
- Do not skip the chapter format structure to save space.
- Do not assume the user is more advanced than the calibration above.
- Do not use em dashes (the user dislikes them stylistically).
- Do not use emoji in the guide.
- Do not generate ahead — one chapter at a time, always.
- Do not commit code on the user's behalf. The user does all commits.
- Do not modify Naitei (the other repo) — this project is naitei-embeddings only. The TypeScript integration is Extra Chapter 1 and is opt-in.

---

## First Action

When the user first invokes you with this memory:

1. Acknowledge that you've read PROJECT.md.
2. Confirm the plan in one short paragraph (what you're about to build).
3. Ask the user to confirm before generating Chapter 0, OR generate Chapter 0 directly if the user has already asked for it.

Do NOT generate anything other than Chapter 0 in the first action. Do NOT pre-create files in `/src`. Chapter 0 itself instructs the user to create the initial files.

---

## End of Memory

That's everything. If anything in here is unclear, the user can clarify. Do not invent or assume beyond what's written here. When in doubt, ask the user.
