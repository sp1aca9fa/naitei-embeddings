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
│   ├── concepts-index.md   (running ledger: concept/tool -> chapter it was introduced in)
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

## Quick recall check (mandatory from Chapter 3 onward)

[Only include once at least one earlier chapter sits 2+ chapters back. 3 short plain-recall questions about concepts from those earlier chapters — not new material, not concepts this chapter is about to teach. Pull from `guide/concepts-index.md` to pick genuinely non-adjacent concepts (favor ones this chapter does NOT already build directly on top of, so it's a real cross-reference and not the same reuse the chapter needed anyway). Put answers behind a collapsible block so he can self-check without spoiling it:

1. [question]
2. [question]
3. [question]

<details>
<summary>Check your answers</summary>

[short answers, 1-2 sentences each]

</details>

If any felt shaky, tell him to skim the referenced chapter before continuing — do not re-teach it here. This is a temperature check, not a gate; he proceeds regardless of how he did.]

## Why this matters

[Theoretical / contextual: why is this concept important, where does it fit in the larger project, what real-world systems use it. Aim for the user to understand the WHY before the HOW.]

## Concepts

### [Concept 1]

[Explain the concept. If it has a mathematical or technical foundation, explain enough of it that the user understands the intuition. Use analogies where helpful. Do not assume prior knowledge beyond CS50P-level Python.]

### [Concept 2]

[As above.]

## The tools we're using

Cover EVERY package, library, AND standard-library module the chapter's code uses, including ones that look trivial (for example `logging`, `pathlib`, `hashlib`). The user's base is CS50P only: he has barely used anything beyond core Python builtins, so he has no working reference for even "simple" stdlib modules and would otherwise have to look each one up online. Bundle that reference into the guide so he has the tools to write everything himself.

For each one:

- What it is: [one sentence]
- What we're trying to achieve with it: [what concrete outcome, in plain language, before any code — e.g. "validate the shape of incoming JSON so bad requests fail fast with a clear error, instead of crashing deep in our match logic"]
- Why it's needed for this project specifically: [not just what the tool does in general, but why THIS project needs it, tied to features A/B or the architecture]
- Quick comparison with alternatives: [1-3 sentences on what else could have been used and why this one was picked, e.g. "FastAPI vs Flask: Flask needs a separate library bolted on for validation; FastAPI bakes it in via Pydantic" — this is what lets him picture the tradeoff, not just accept a fait accompli]
- How it fits into our code: [where in the request/response/data flow this tool's output plugs in, described in prose — what feeds it, what it hands off to next]
- Beyond this project: describe at least 3 realistic use cases for this tool/package that have nothing to do with naitei-embeddings — common industry usage, other problems it's known for solving. Skip a use case only if it's genuinely no longer relevant (deprecated pattern, etc.), don't pad to hit 3. Also briefly cover other major functionality/methods the package offers beyond what this chapter actually uses, so the user has a working sense of the tool's full surface, not just the slice this project needed.
- Install: `pip install ...` (or note it's stdlib, no install)
- Docs: [link]
- Key methods/classes we'll use, each with a SMALL, SELF-CONTAINED, RUNNABLE usage example that shows the actual call and its output, not just a prose description:
  - `method_or_class_name(...)` — [what it does, with a runnable code snippet and its input → output]
  - ...

The goal: after reading this section he should already be able to picture what the exercise is going to have him build and why, before he opens the exercise. He should never reach the exercise instructions confused about what the tool is for or how it connects to the project; the exercise should feel like "apply what I now understand" rather than "the first place I understood this." The runnable examples still need to teach enough that the matching exercise can apply the tool with a slight twist (so the exercise is not a copy-paste of the example) while still giving him everything he needs to solve it. Where a module is so trivial that the explanation effectively IS the exercise usage, accept that overlap but keep the example minimal.

**Completeness check (mandatory, do this before finalizing the chapter):** walk through the exercise's expected solution step by step, and confirm every operation it requires was actually taught somewhere in "Concepts," "The tools we're using," or "How it fits together" — not just the main library calls, but manipulation of their output too (e.g. if the exercise needs per-row or per-column access into a matrix/array, that indexing operation must be explicitly shown with a runnable example before the exercise, not left implicit in a one-line mention). If a needed operation isn't covered yet, add a short concept/tool entry for it rather than relying on the exercise or hints to introduce it for the first time. The chapter's job is to make sure he HAS every tool needed to reach the solution himself; the exercise's job is to make him assemble them, not to discover a missing one.

### [Package or library 2]

[As above.]

## How it fits together

[Walk through what the code in this chapter does at a high level. Diagrams in ASCII or markdown are welcome. Show the data flow.]

## Code examples

[2-4 small, focused code examples illustrating the concepts. These are NOT the user's solution. They demonstrate what the tools do. Each example should have a short "input → output" demonstration.

Minimize handed-out code, even for new concepts. Keep these examples at the concept level on throwaway/toy structures (a fake `items` table, an `Animal` class, etc.) rather than on the chapter's actual project files, so the user still has to write the real code himself in the exercises. Push fuller, closer-to-the-solution skeletons down into the per-exercise collapsible Hints blocks, where they serve as a safety net the user opts into, rather than the first thing he reads.

Occasionally, when a natural candidate exists (not every chapter, use judgment), one example may show a subtly buggy "before" snippet on a toy structure and ask him to spot the bug before revealing the fix, rather than presenting only correct code. Debugging existing code is a distinct skill from writing new code and he'll do far more of the former in real work. Don't force this if nothing in the chapter lends itself to it naturally.]

## Your tasks

Every core chapter has three hands-on coding exercises, plus a 4th retrieval-practice exercise from Chapter 3 onward (see "Retrieval callback" below), plus a 5th always-optional broader-exploration exercise (see "Beyond this project" below). Exercises 1 and 2 are required; Exercise 3 is ALWAYS included (always authored) but is optional for the user to complete, so he can skip it to keep momentum. Exercise 4, when present, is required (it's short by design, so skipping it isn't necessary). Exercise 5 is always optional to complete. Rules:

- **Anchor every exercise to the chapter's CORE skill** (the concrete competency the chapter teaches; e.g. Chapter 4's core skill is "database operations with vectors"). Exercises drill that skill, and any divergence from it should be MEASURED. An exercise that practices the core skill on a throwaway (e.g. "delete a vector, then re-query to see it gone") is on-target even if it does not advance the product. An exercise that wanders into a tangential tool/concept (e.g. an SQL-composition refactor inside a DB-vectors chapter) is off-target: it is a FALLBACK, used only when there is no core-skill practice left to assign, never the default. Concretely, the Chapter 4 sequence should have felt like insert (Ex1) -> search (Ex2) -> another vector DB operation like delete/update (Ex3), not insert -> search -> SQL refactor.
- **At least one exercise per chapter must advance the actual project** (features A/B / the FastAPI service), so the guide keeps shipping real functionality.
- **Beyond that anchor, exercises MAY practice concepts taught in the chapter without building the main feature.** The guide is encouraged to teach useful/adjacent concepts that features A/B do not strictly require (this deepens his learning and makes the exercises more dynamic), and an exercise may drill one of those concepts, so long as it stays within a measured distance of the chapter's core skill per the anchor rule above.
- Exercise 1 and Exercise 2 must both be the user writing code that directly applies what the chapter just taught. Testing is NOT a default exercise; it is only one option for the optional third slot.
- The user writes the code himself in `/src`, `/scripts`, or wherever the chapter directs. Throwaway exploration scripts under `/scripts` are kept as a record of the learning process.
- **Exercise instructions describe requirements in prose, never in literal code.** State what a field represents and its constraint in words ("the request needs the CV bullets as a list of strings, and the JD's language, defaulting to English"), not as a signature he can copy-paste ("`cv_bullets: list[str]`", "`jd_language: str = \"en\"`"). Same for function signatures, class shapes, endpoint decorators: describe the behavior and constraints, not the syntax. This applies to the exercise body itself. It does NOT apply to the "Stuck? Hints" block, where an actual code skeleton is expected as the final, most-revealing nudge — that is the intended safety net, and the point is he has to open it to see real code instead of finding it already in the instructions.

Each exercise carries its OWN pitfalls and hints inline (structure below), so the safety net is attached to the task instead of buried at the end of the chapter.

**Difficulty reference (mandatory calibration for every chapter).** A "gap" is a point in the exercise's solution that the chapter's guide does NOT hand him directly — a step he must reason out himself rather than copy the shape of from an example.

- **Exercise 1 (Fixate):** follows the shape of what the chapter's examples already demonstrated — one tool, one pattern, applied directly, not combined with others in a new way. Contains exactly ~2 gaps: small stretches not explicitly shown (a check, an edge case, a minor transformation). Overall feel: easy, with two moments to stop and think.
- **Exercise 2 (Apply):** requires combining 2+ concepts/tools from the chapter in a way the chapter's own examples didn't already show combined. Contains ~3 gaps, each sitting at a seam between the combined concepts rather than inside a single pattern. Overall feel: noticeably harder than Exercise 1, because the combining itself is the difficulty, not just the individual gaps.
- **Exercise 3 (Apply, optional to complete):** same shape as Exercise 2 (combine 2+ concepts, ~3 gaps), pitched slightly above it — either one more combination, or gaps that require noticing something (an edge case, a tradeoff) rather than just filling in a step.

**Beyond this project (always-optional 5th exercise).** Draw on the "Beyond this project" use cases from "The tools we're using" to set an exercise on a toy/generic scenario that has nothing to do with naitei-embeddings — something common in industry or useful for his career generally (e.g. if the chapter covers `hashlib`, an exercise might be deduplicating a list of uploaded files by content hash, unrelated to embeddings). This exists purely to broaden exposure to a tool beyond the one slice this project needs; skip it for a chapter if none of its tools have a use case interesting/distinct enough to justify a whole exercise (don't force one). Same "Pitfalls"/"Stuck? Hints" structure as the other exercises, but lighter — this is exploration, not core-skill drilling, so 1-2 gaps at most.

**Retrieval callback (mandatory 4th exercise, from the first chapter that has an earlier chapter sitting 2+ chapters back — i.e. starting at Chapter 3).** Every such chapter has a short, always-required Exercise 4 that makes the user reuse a concept or tool from a non-adjacent earlier chapter (2+ chapters back), with NO refresher text reminding him how it works — just a one-line pointer to which chapter it came from (e.g. "recall Chapter 2's point about X — not re-explained here"). This is retrieval practice: recalling something without it being re-taught is what makes it stick long-term, distinct from Exercises 2/3 which combine THIS chapter's material. Keep it deliberately small: a few lines of code, or a short written answer, never a new feature, and never gapped the way Exercises 1-3 are (0-1 gap at most — this exercise tests memory, not problem-solving). Prefer a chapter/concept the current chapter does NOT already build directly on top of, so it's a genuine cross-reference rather than the reuse the chapter is already doing for its core work. Give it its own short pitfalls line if relevant, but hints are usually unnecessary since the point is unaided recall; if a hint is warranted, it should only be a pointer back to which chapter/section to skim, never the answer itself.

### Exercise 1: Replication

[A first exercise where the user writes code similar to what the chapter just taught. Spell out:
- What files to create or modify
- What functions/classes to write
- What behavior is expected
- How to verify it works (a one-line test, or a curl command, or similar)]

**Pitfalls for this exercise:** [1-3 gotchas specific to THIS exercise, and how to fix each. Be specific.]

<details>
<summary>Stuck? Hints (click to expand)</summary>

Progressive hints for THIS exercise, gentlest first. Keep the depth proportional to the exercise; a simple one may need only the first nudge.
- Tool-usage guidance (NO pseudocode, NO code): plain-text walkthrough of which tools/functions from this chapter to use and in what order, e.g. "split the text into sentences with the sentence-splitting library, embed the two groups separately, compare them the way the chapter showed, then figure out how to look at one requirement's scores at a time before deciding what counts as a good match." This should point at the shape of the solution (including any step the "Concepts" or "Tools" sections might not have made obvious was needed, like operating on a single row/column at a time) without naming variables, functions, or syntax.
- Approach / pseudocode: now give pseudocode, but keep it deliberately looser than the real code, not a near-transliteration of it. It should read like a rough sketch a person would jot on paper (e.g. "for each requirement: grab its scores, find the best ones, check if any is good enough"), not something with real function names, real data structures, or a shape that maps 1:1 onto the actual implementation. It must still clearly lead to the correct solution structure, including any control flow (loops, conditionals) the exercise actually needs; do not omit a step just because it's easy to forget to mention.
- Code skeleton: actual code with one or two key parts blanked out, or a full solution if the earlier nudges were not enough. The safety net.

</details>

### Exercise 2: Application

[A second exercise where the user uses the chapter's core skill to produce an intermediate result and then uses that result to reach a final answer. Still hands-on coding of the chapter's core skill, not a test-writing exercise. Same level of detail as Exercise 1.]

**Pitfalls for this exercise:** [as above.]

<details>
<summary>Stuck? Hints (click to expand)</summary>

[Progressive hints for THIS exercise, same structure as Exercise 1.]

</details>

### Exercise 3 (optional to complete): extension on the chapter's core skill

[ALWAYS author this exercise; do NOT omit it. Mark it clearly as optional to COMPLETE (the user may skip it to make progress), but it must always be present. PREFER an exercise that drills the chapter's core skill with a new operation or variation (e.g. in a DB-vectors chapter: delete or update a vector, then re-query to confirm the effect), because that reinforces the chapter's actual competency. Use these alternatives, in descending preference, only when no natural core-skill extension exists: practice of an adjacent concept the chapter taught; or a test-writing exercise (only when the chapter produced something testable without external dependencies). There should always be at least one workable option, so the exercise is always included.]

**Pitfalls for this exercise:** [as above.]

<details>
<summary>Stuck? Hints (click to expand)</summary>

[Progressive hints for THIS exercise, same structure as Exercise 1.]

</details>

### Exercise 4 (retrieval, from Chapter 3 onward): recall a non-adjacent earlier concept

[Only present from the first chapter that has an earlier chapter 2+ chapters back. A short task requiring the user to reuse a concept/tool from a non-adjacent earlier chapter, named only by a one-line pointer ("recall Chapter N's point about X") with NO refresher of how it works. Keep it small: a few lines of code or a short written answer, never a new feature. Do not gap it like Exercises 1-3; this tests memory, not problem-solving.]

**Pitfalls for this exercise:** [optional, only if there's a genuine common mistake.]

<details>
<summary>Stuck? Hints (click to expand)</summary>

[Usually just a pointer back to which chapter/section to skim. Never give the answer directly here; that would defeat the point of unaided recall.]

</details>

### Exercise 5 (optional to complete): beyond this project

[Only include when at least one tool from this chapter has a use case worth exploring outside naitei-embeddings (see "Beyond this project" above). A short exercise on a generic/toy scenario, unrelated to the CV/JD matching product, that exercises the tool/package in a different, industry-common way. Omit this section entirely for chapters where nothing warrants it — don't force it.]

**Pitfalls for this exercise:** [as above, optional.]

<details>
<summary>Stuck? Hints (click to expand)</summary>

[Progressive hints for THIS exercise, same structure as Exercise 1, but lighter — 1-2 gaps at most.]

</details>

## Common pitfalls (chapter-wide, optional)

[Only cross-cutting gotchas NOT tied to a single exercise (e.g. "always call `register_vector` on every connection", "remember to `commit` after writes"). Keep it short, or omit the section entirely when every pitfall already lives under an exercise.]

## Explain it back (mandatory, every chapter)

[One prompt asking him to write a few sentences, in his own words, no code, explaining WHY a design choice in this chapter was made, not how the code works (e.g. "why a compound unique index instead of a single hash column?"). Verbalizing without code exercises a different kind of recall than writing code and catches surface-level "I copied the shape but don't know why" understanding. Keep it to one question; this is not a written exam.]

## Further reading

[2-4 links: official docs, relevant blog posts, deeper explorations. Optional reading, not required to proceed.]

## Checkpoint

Before moving to Chapter N+1, you should have:
- [ ] [Concrete deliverable from Exercise 1]
- [ ] [Concrete deliverable from Exercise 2]
- [ ] [Concrete deliverable from Exercise 4, when present (retrieval callback)]
- [ ] [Concrete deliverable from Exercise 5, when present (beyond this project)]
- [ ] [Any conceptual takeaways the user should have internalized]
- [ ] Written answer to the "Explain it back" prompt
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

4. **Retrieval check lives in the chapter itself, not in chat.** From Chapter 3 onward, every generated chapter opens with a "Quick recall check" section (see Chapter Format) — 3 plain-recall questions about concepts from 2+ chapters back, with self-check answers behind a collapsible block. You do not need to ask these in chat before generating; picking good questions is part of writing the chapter. Consult `guide/concepts-index.md` when choosing which older concepts to draw from.

5. **Never generate more than one chapter at a time.** Even if the user asks for "all chapters" or "the next two chapters", politely refuse and explain that the workflow is one at a time, because each chapter depends on the previous chapter's actual code.

6. **Never get ahead.** If a chapter introduces concept X and a later chapter introduces concept Y that builds on X, don't sneak Y into the X chapter. Each chapter has one focus.

7. **Track progress at the end of each chapter.** End each chapter with the Checkpoint section listing what the user should have. This is also useful for verification in the next chapter.

8. **Update `guide/concepts-index.md` after finalizing each chapter.** Append one line per new concept/tool the chapter introduced: `- [Concept/tool name] — Chapter N`. This is the lookup you use to pick genuinely non-adjacent material for the next chapter's "Quick recall check" and Exercise 4 retrieval callback, instead of re-skimming every prior chapter from scratch each time.

9. **Written answers (Exercise 4 recall, "Explain it back" prompt) go in `guide/answers.md`.** One running file, appended to as chapters are completed, one section per chapter (`## Chapter N`). Do not create a new file per chapter for this.

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

**Deepening his Python is a co-equal goal of this project, not a side effect.** He is here to learn ML/embeddings AND to become a stronger Python developer. CS50P gave him core language basics (variables, functions, conditionals, loops, exceptions, file I/O, regex, basic OOP, pytest, basic Flask) but did NOT teach how real Python projects are built: packaging and `__init__.py`, imports (absolute vs relative), module-vs-script execution, virtualenvs, the type system, decorators, context managers, generators, async, dunder methods, etc. Treat every such concept the chapter's code touches as something to TEACH with a short focused explanation the first time it appears, not to use silently. He explicitly wants these explained (his `__init__.py` question is a representative example). This is separate from, and in addition to, the "explain every package/module with runnable examples" rule in the chapter-format sections above.

When explaining things:
- Don't over-explain the CS50P basics he already knows (loops, functions, OOP fundamentals).
- DO teach Python-specific idioms the first time they appear (decorators, context managers, generators, async, comprehensions used non-obviously, dunder methods, the type-hint system beyond trivial annotations). A few sentences on what it is and why it's used, tied to the concrete line of code in front of him.
- DO teach non-basic Python ecosystem/project-structure mechanics when the chapter touches them, even when they are not the chapter's ML topic: imports (absolute vs relative), packages and `__init__.py`, circular imports, `sys.path` and script-vs-module execution (`python -m`), virtualenvs, `requirements.txt`, etc. He got stuck on multi-file structure and imports in Chapter 3 (his first Python app spanning more than two files). Flag the likely gotcha proactively rather than waiting for him to trip on it.
- Keep each explanation tight and anchored to the code at hand; the guide should read as "here's the Python concept this line relies on," not as a standalone Python textbook. Breadth comes from covering concepts as they arise across chapters, not from long digressions in any one chapter.
- DO explain every new library and every new concept the first time it appears. "Explain" means SHOW a runnable usage example with its output, not just describe it in prose. This applies even to simple standard-library modules (`logging`, `pathlib`, etc.); he has no prior reference for them and would otherwise have to look them up online anyway, so bundle that reference into the guide. See the "The tools we're using" section for the required format.
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
