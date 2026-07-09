# Chapter 7: Skill Canonicalization

## What we're building in this chapter

A `POST /canonicalize-skills` endpoint. It takes a list of skill mentions in any phrasing or language ("React", "Reactを使用", "リアクト", "TS") and, for each one, returns the closest match from a fixed list of canonical skill names, along with a similarity score, or nothing if no canonical skill is actually close enough to be trusted as a match.

## Quick recall check

Before diving in, answer these without looking back. They're not new material, just a check that older concepts are still solid.

1. Chapter 0: what problem does an ASGI server like `uvicorn` solve that a plain Python script running your FastAPI app object wouldn't?
2. Chapter 3: why does `MockProvider` hash the input text to seed its random generator, instead of just calling `np.random.default_rng()` with no seed at all?
3. Chapter 4: why can't you store vectors from two different embedding models in the same column and treat them as comparable?

<details>
<summary>Check your answers</summary>

1. Your FastAPI app object only *describes* routes and handlers; nothing about it listens on a network port, parses raw HTTP/ASGI traffic, or manages concurrent connections. `uvicorn` is the server process that actually does that and hands parsed requests to your app.
2. Seeding with a hash of the input text makes the output deterministic: the same string always produces the same fake vector, every run, on any machine. An unseeded generator gives a different vector every call, which breaks the "same text should hit the cache" behavior tests and the caching logic both depend on.
3. Different models place vectors in different, unrelated coordinate systems (and often different dimensions). A vector from one model and a vector from another aren't expressing "distance" in the same space, so comparing them (or averaging them, or storing them as if interchangeable) is meaningless, not just imprecise.

</details>

If any felt shaky, skim the referenced chapter before continuing.

## Why this matters

Naitei's users describe the same skill in wildly different ways depending on where the text came from: a JD in Japanese says "リアクト" or "Reactエンジニア経験者", a CV bullet the user wrote says "React", an old CV version says "ReactJS". If Naitei's skill tracking treats each spelling as a distinct skill, every feature built on top of it, coverage gaps, skill history over time, "skills you're missing for this role", fragments into noise. The fix isn't a bigger dictionary of exact-match aliases (that list never stops growing and breaks the moment someone phrases something in a way you didn't anticipate); it's the same tool this whole project already relies on: if two skill mentions embed close together, they probably mean the same thing, regardless of language or exact wording.

This pattern is not specific to job-hunting. E-commerce catalogs canonicalize "sneaker", "trainers", and "スニーカー" to one product category so search and recommendations don't fragment across spellings. Master data management (MDM) systems reconcile "IBM", "International Business Machines", and "I.B.M." into one company record. Search engines normalize query variants ("NYC" and "New York City") before hitting an index. Every one of these is the same shape of problem you're solving here: given a fixed, known set of "real" entities and a new, messy mention, decide which real entity it refers to, or admit you don't know.

This chapter also reuses something you already built in Chapter 2's minimal nearest-neighbor demo, but for real this time: a permanent reference list instead of two throwaway words, wrapped in an actual endpoint, with a real decision about when to say "I don't know" instead of guessing.

## Concepts

### Nearest-neighbor lookup against a fixed reference set

Chapter 6's cache answered "have I seen this exact text before?" against a set of entries that started empty and grew forever. This chapter's problem is different in shape: you have a small, fixed list of canonical skill names, known in advance, that doesn't grow while the service is running. Given a new skill mention, you're not asking "have I seen this exact text?", you're asking "which of my known items is this new thing closest to in meaning?"

That's a nearest-neighbor lookup: embed the unknown item, compare its vector against every reference vector, and take whichever reference is closest. On its own, this always returns *something* — even a completely unrelated string still has some closest reference, just a distant one. A canonicalizer that always answers, even when nothing is actually close, will confidently mislabel real gaps in your canonical list as matches. The fix is a rejection option: if even the *closest* reference isn't close enough, the answer is "no confident match" rather than a forced guess. That's the difference between a system that's occasionally wrong in a way you can measure, and one that's silently, confidently wrong.

### Precompute once, reuse for the life of the process

Chapter 6 built a cache that lives in Postgres because the set of things it needs to remember (every CV bullet and JD sentence ever seen) is unbounded and grows across restarts. The canonical skill list is the opposite kind of data: a few hundred fixed strings that don't change while the service runs, and don't need to survive being recomputed on the next deploy. There's no reason to hit a database for this at all. Instead, you embed the entire canonical list exactly once, when the process starts (in practice: once, when your module is first imported, the same moment `provider = get_provider()` already runs today), and keep the resulting matrix sitting in memory for as long as the process lives. Every request after that reuses the same in-memory matrix; the canonical side of this feature never touches the database.

This is the same "don't repeat expensive work" instinct behind Chapter 6's caching, but the shape of the data tells you the right mechanism is different: a growing, unpredictable set needs persistent storage and a lookup key; a small, static set just needs to be computed once and held in a variable.

### Threshold tuning as a tradeoff, not a fixed constant

A good multilingual embedding model tends to put genuinely related pairs noticeably closer together than unrelated pairs, but there is no universal similarity score that means "same skill" across every model and every domain, the same way there's no universal "this email is spam" score that works for every inbox. Set your threshold too low, and things that aren't really the same skill still get force-matched ("React Native" gets folded into "React", quietly losing information). Set it too high, and real matches get rejected as unknown ("Reactを使用" fails to match "React" purely because of phrasing). There isn't a table you can look up for "the right number for `multilingual-e5-base`" — you have to run a handful of skill mentions you already know the right answer for, look at the scores that come back, and pick a threshold that separates the good matches from the bad ones for *your* model and *your* list. Expect to revisit the number after looking at real output, not to get it right on the first guess.

One thing worth noticing about this task specifically, that Chapter 6 didn't have: matching a skill mention against a canonical skill name is a *symmetric* comparison, not a search. You're not asking "does this piece of text satisfy that requirement" (the retrieval framing from `/match`, where JD requirements search over CV bullets); you're asking "do these two strings refer to the same thing", which is true or false in both directions equally. If you looked back at Chapter 2's `query:` / `passage:` prefix question and (correctly) concluded that `/match` needs different prefixes on its two asymmetric sides, this chapter is the other case: since neither side is "searching" the other, both the canonical skill names and the incoming skill mentions should get the same prefix treatment. Worth sitting with the contrast between the two chapters rather than assuming one rule applies everywhere.

## The tools we're using

### `json` (standard library)

- What it is: Python's standard-library module for reading and writing JSON, the text format almost every web API and config file uses to represent structured data.
- What we're trying to achieve with it: your canonical skill list lives in a file as a JSON array of strings; you need to turn that file's text content into an actual Python list you can loop over and pass to your provider.
- Why it's needed for this project specifically: the canonical list is data, not code, and it needs to be editable without touching Python (adding a skill next month should mean editing a JSON file, not a source file). `json` is what turns that file's contents into Python objects your embedding code can use.
- Quick comparison with alternatives: you could store the list as a plain Python list literal directly in a `.py` file, and that would technically work. The reason not to: mixing a growing, frequently-edited data list into source code makes diffs noisier and tempts you to write logic in the same file as the data. A `.json` file keeps "the list of skills" and "the code that uses it" cleanly separate, and it's the same format you'll eventually see coming back from real HTTP APIs, so the pattern transfers.
- How it fits into our code: read once, at import time, alongside where `provider = get_provider()` already runs — the loaded list feeds straight into a batch `provider.embed(...)` call, and the result is held in memory for every request that follows.
- Beyond this project: (1) reading and writing application configuration files, feature flags, and per-environment settings that need to be edited without a code deploy; (2) parsing responses from REST APIs, since the overwhelming majority speak JSON, so `json.loads` (or a library that wraps it, like `httpx`'s `.json()` method) is something you'll use constantly working with any external service; (3) caching an expensive computation's result to disk between runs of a script, so a second run can `json.load` a prior result instead of recomputing it, a very common pattern in data-processing and ML pipelines specifically. Beyond what this chapter uses: `json.dump(obj, file)` / `json.dumps(obj)` write Python objects back out as JSON text (the `indent=2` keyword argument pretty-prints it for humans instead of producing one dense line); `json.JSONDecodeError` is the exception raised on malformed input, worth catching explicitly around any JSON you didn't generate yourself; the `default=` keyword argument on `dumps` lets you tell `json` how to serialize objects it doesn't know natively (like a `datetime` or a custom class).
- Install: standard library, no install.
- Docs: <https://docs.python.org/3/library/json.html>
- Key functions we'll use, with a runnable example:
  - `json.loads(s: str)` — parses a JSON-formatted string into Python objects (a JSON array becomes a `list`, a JSON object becomes a `dict`).

```python
import json
from pathlib import Path

Path("fruits.json").write_text('["apple", "banana", "cherry"]')

raw = Path("fruits.json").read_text()
fruits = json.loads(raw)
print(fruits)
print(type(fruits))
```

```
['apple', 'banana', 'cherry']
<class 'list'>
```

You've already used `Path(...).read_text()` (or the equivalent via `open`) since Chapter 0's `.env` loading; `json.loads` just takes the resulting string one step further, turning JSON text into real Python data instead of leaving it as one big string you'd have to parse by hand.

### NumPy: comparing one vector against many, and finding the best match

You've used `@` for a full batch of vectors against another full batch (`bullet_vecs @ req_vecs.T`) since Chapter 5. This chapter needs two related, smaller operations you haven't used yet.

**One vector against a matrix of many.** When you're canonicalizing a single skill mention, you have one embedding (1D) and a matrix of canonical embeddings (2D, one row per canonical skill). `@` still works here: a 1D vector times a 2D matrix's transpose gives you back a 1D array with one similarity score per canonical skill.

```python
import numpy as np

skill_vec = np.array([1.0, 0.0, 0.0])
canonical_matrix = np.array([
    [1.0, 0.0, 0.0],   # "React"
    [0.0, 1.0, 0.0],   # "Django"
    [0.7, 0.7, 0.0],   # "Vue"
])

scores = canonical_matrix @ skill_vec
print(scores)
```

```
[1.  0.  0.7]
```

One score per row of `canonical_matrix`, in the same order the canonical list was in.

**Finding the index of the best score.** `np.argsort` (Chapter 5) gives you a full ranking; here you only want the single best match, which is what `np.argmax` is for.

```python
best_index = np.argmax(scores)
print(best_index)
print(scores[best_index])
```

```
0
1.0
```

**The same thing, but row-by-row across a whole matrix at once.** If you have several skill mentions embedded together as a 2D matrix (`mentions @ canonical_matrix.T` gives you a 2D grid: one row per mention, one column per canonical skill), you want the best match *per row*, not one single best match across the whole grid. Passing `axis=1` tells `argmax` to find the best column *within each row* independently, returning one index per row instead of one index overall.

```python
grid = np.array([
    [1.0, 0.0, 0.7],   # mention 1's scores against each canonical skill
    [0.2, 0.9, 0.1],   # mention 2's scores
])

best_per_row = np.argmax(grid, axis=1)
print(best_per_row)
```

```
[0 1]
```

Mention 1's best match is canonical index 0; mention 2's best match is canonical index 1. Without `axis=1`, `np.argmax(grid)` would flatten the whole grid and give you a single index into the flattened array, which is essentially never what you want when you have more than one row to score independently.

## How it fits together

```
   guide/answers.md, data/canonical_skills.json
                    |
        (once, at module import, next to `provider = get_provider()`)
                    |
                    v
   canonical_skills = json.loads(Path(...).read_text())
   canonical_matrix  = provider.embed(canonical_skills)      <- held in memory
                    |
                    |
   POST /canonicalize-skills  { skills: [...] }
                    |
                    v
   for each skill mention:
        vec = provider.embed([skill])[0]
        scores = canonical_matrix @ vec
        best_index = np.argmax(scores)
        best_score = scores[best_index]
                    |
          +---------+---------+
          | best_score >= threshold      | best_score < threshold
          v                              v
   canonical_skills[best_index], score      None, log the rejected mention
                    |
                    v
        collect one result per input skill, same order
                    |
                    v
        return the list as the response
```

Nothing here touches `db.py` or Postgres. The canonical side is computed once and lives in memory for the process's whole lifetime; the request side is embedded fresh every call, same as `/match` was in Chapter 5 before caching existed.

## Code examples

These work against a throwaway reference list, not your real canonical skill list, so you still write the actual feature yourself.

### Example 1: load a small JSON list and embed it once

```python
import json
from pathlib import Path
import numpy as np

Path("colors.json").write_text(json.dumps(["red", "green", "blue"]))

colors = json.loads(Path("colors.json").read_text())

def fake_embed(words: list[str]) -> np.ndarray:
    # stand-in for provider.embed(); real vectors would come from your provider
    lookup = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}
    return np.array([lookup[w] for w in words], dtype=float)

color_matrix = fake_embed(colors)
print(color_matrix.shape)
```

```
(3, 3)
```

### Example 2: nearest match, with a threshold that can say no

```python
import numpy as np

reference_names = ["red", "green", "blue"]
reference_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

def closest_match(vec: np.ndarray, threshold: float) -> tuple[str | None, float]:
    scores = reference_matrix @ vec
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    if best_score < threshold:
        return None, best_score
    return reference_names[best_index], best_score

print(closest_match(np.array([0.9, 0.1, 0.0]), threshold=0.5))
print(closest_match(np.array([0.3, 0.3, 0.3]), threshold=0.5))
```

```
('red', 0.9)
(None, 0.3)
```

The second call's best score (0.3, against every reference equally, since the input is equidistant from all three) still isn't close to any single reference, so the threshold rejects it instead of forcing a pick.

### Example 3: spot the bug

```python
def closest_match_buggy(vec, reference_matrix, reference_names, threshold):
    scores = reference_matrix @ vec
    best_index = np.argmax(scores)
    return reference_names[best_index], scores[best_index]
```

<details>
<summary>What's wrong with it?</summary>

It never checks the threshold at all, it always returns the best available match no matter how low the score, even for input that has nothing to do with any reference item. The `threshold` parameter is accepted but never used. This is exactly the "always answers, never admits it doesn't know" failure mode the concepts section warned about.

</details>

## Your tasks

### Exercise 1 (Fixate): match one skill mention against the canonical list

Write the core lookup, on its own, before it's wired into an endpoint.

**What to do:**

1. Somewhere in your project (`src/db.py` is the wrong place, since this doesn't touch the database; a new module or directly in `main.py` both work), write a function that takes a single skill mention, the list of canonical skill names, their already-computed embedding matrix, and your provider, and returns the closest canonical skill name and its similarity score, without any threshold logic yet, just the raw best match.
2. As a throwaway check (a scratch script, not a permanent test), pick 4-5 short strings by hand (a couple of skill names, a couple of skill names in a different phrasing or language, one unrelated word) and call your function against a small hand-written canonical list to confirm the scores look like what you'd expect before wiring anything bigger together.

**How to verify it works:** the unrelated word should score noticeably lower than the closer matches. If everything scores about the same regardless of input, something's wrong before you add the threshold at all.

**Pitfalls for this exercise:** `provider.embed([skill])` returns a 2D array even for one string, same shape gotcha as Chapters 5 and 6, index into row 0 before doing anything else with it; make sure the canonical matrix's row order and the canonical name list's order stay in lockstep, since `np.argmax` only gives you back a numeric index, and that index means nothing if the two lists have drifted out of sync.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** Embed the single mention the same way you'd embed anything with your provider, then remember it comes back 2D and needs indexing down to 1D. Multiply that 1D vector against the canonical matrix the way the tools section showed (`matrix @ vec`) to get one score per canonical skill, then use `np.argmax` to find which position scored highest, and use that same position to index into both the score array and the canonical name list.

**Approach / pseudocode:**
```
to find the closest canonical skill for one mention:
    embed the mention, reduce it to a single vector
    score it against every canonical skill at once
    find the position of the highest score
    look up the canonical name and score at that position
    return both
```

**Code skeleton:**
```python
def closest_canonical_skill(skill: str, canonical_skills: list[str], canonical_matrix: np.ndarray, provider) -> tuple[str, float]:
    """Return the closest canonical skill name and its similarity score, no thresholding."""
    vec = provider.embed([skill])[0]
    scores = ...  # canonical_matrix @ vec
    best_index = ...  # np.argmax
    return canonical_skills[best_index], scores[best_index]
```

</details>

### Exercise 2 (Apply): the real endpoint, with a threshold

Turn Exercise 1's function into `POST /canonicalize-skills`.

**What to do:**

1. Create a starter canonical skill list as a JSON file (see the list provided below), and, near where `provider = get_provider()` already runs in `main.py`, load that file and embed the whole list once into a matrix held at module level.
2. Design the request model: it needs a list of skill mention strings. Design the response model: a list of results, one per input skill, each carrying the original mention, the matched canonical skill (which needs to be able to be absent, not just an empty string, for the no-match case), and the score.
3. Add a way to configure the similarity threshold (an environment variable, following the same pattern `config.py` already uses for other tunable values) and use it to decide, per skill, whether to return the match from Exercise 1's function or an absent match.
4. When a skill mention gets rejected as unmatched, log it (not just silently drop it) so an unmatched skill is something you could review and potentially add to the canonical list later, rather than a mention that just vanishes.

**How to verify it works:** hit `/canonicalize-skills` through `/docs` with a mix of exact canonical names, differently-phrased or Japanese versions of skills you know are in your list, and at least one string that shouldn't match anything ("purple giraffe farming" or similar). Confirm the clear matches come back with high scores, the paraphrased ones still match (tune your threshold if they don't), and the nonsense one comes back with no match.

**Pitfalls for this exercise:** re-loading the JSON file or re-embedding the canonical list inside the request handler instead of once at import time defeats the entire "precompute once" point of this chapter and makes every request pay the canonical-list embedding cost again; a field that's sometimes a string and sometimes missing needs a type that says so explicitly (you've already written this exact shape once, in Chapter 4/6's `db.py`, where a lookup function's return type says it can come back empty); picking a threshold without actually testing a few known-good and known-bad pairs first and just guessing a round number.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** The canonical list loading and embedding belongs at the same level as `provider = get_provider()`, run once when the module is imported, not inside any function. For the response model's nullable field, look at how you've already typed a value that can be "found" or "not found" elsewhere in this project, the same idea applies to a Pydantic model field, not just a function's return type. Read the threshold from an environment variable the same way `config.py` already reads `MOCK_DIMENSION` or similar, as a number, with a default. Inside the endpoint, loop over the incoming skill list, call Exercise 1's function once per mention, and decide match-or-not by comparing its score against the threshold before building each result.

**Approach / pseudocode:**
```
at module load time:
    read the canonical skill list from its JSON file
    embed the whole list once, keep the matrix around

for each request:
    for each skill mention in the request:
        find its closest canonical skill and score, using Exercise 1's function
        if the score clears the threshold: keep the match
        else: log it, record no match
    return one result per mention, in the same order they came in
```

</details>

### Exercise 3 (optional to complete): batch the whole request at once

Same feature, one core-skill variation: stop calling the embedding model once per skill mention.

**What to do:** instead of looping over the request's skill list and calling Exercise 1's function (and therefore `provider.embed`) once per mention, embed the entire incoming list in a single batched call, the way `/match` already batches CV bullets. Build the full mentions-by-canonical-skills similarity grid in one matrix multiply, and use `np.argmax` with the row-wise `axis` argument from the tools section to find each mention's best match in one shot, instead of one `argmax` call per mention.

**Pitfalls for this exercise:** forgetting the `axis=1` argument gives you the single best score across the *entire* grid, one answer total instead of one per row, which silently produces the wrong shape of result; keep the mapping between a request-list position and a grid row consistent, the same ordering discipline Chapter 6 already made you careful about.

<details>
<summary>Stuck? Hints (click to expand)</summary>

**Tool-usage guidance:** Replace the per-mention loop's embedding call with one `provider.embed(...)` call on the whole list of mentions, giving you a 2D matrix, one row per mention. Multiply that against the canonical matrix's transpose to get the full grid of scores in one step, then use `np.argmax(grid, axis=1)` to get one best-index per row. You still loop once per mention afterward, just to apply the threshold and build each result, that part doesn't go away, only the repeated embedding calls do.

</details>

### Exercise 4 (retrieval): recall Chapter 3's abstract base class

This one is not about canonicalization. It is a quick memory check on something from several chapters back that this chapter does not touch at all.

**What to do:** without re-reading Chapter 3, write down (in `guide/answers.md`) your answer to this: why is `EmbeddingProvider` defined as an abstract base class instead of a plain class, and what actually happens if you try to write `EmbeddingProvider()` directly instead of instantiating one of its subclasses? If you're not sure, try it in a scratch Python shell and see what Python tells you, then explain in your own words why that's the intended behavior, not a bug.

**Pitfalls for this exercise:** none specific.

<details>
<summary>Stuck? Hints (click to expand)</summary>

Skim Chapter 3's section on `abc` and abstract methods, then come back and answer without the page open.

</details>

### Exercise 5 (optional to complete): beyond this project — merging config with defaults

A short exercise on `json`, unrelated to skills or embeddings entirely. This pattern (load a config file that might not specify everything, fall back to sensible defaults for anything it leaves out) shows up in almost every real application you'll touch professionally.

**What to do:** on a throwaway script or scratch file, write a function that takes a file path and a dictionary of default settings, and returns a single merged dictionary: if the file exists, its keys override the defaults; if a key isn't present in the file (or the file doesn't exist at all), the default for that key is used instead. Try it with a small JSON file that only overrides one or two of several default keys, and confirm the merged result has all the keys, with only the overridden ones changed.

**Pitfalls for this exercise:** trying to read a file that doesn't exist will raise an error if you don't check for it first (`Path.exists()` is one way); a naive merge that just does `defaults.update(loaded)` in the wrong direction will let the file's keys silently overwrite ones you meant to keep as defaults, or vice versa, so be deliberate about which side wins.

<details>
<summary>Stuck? Hints (click to expand)</summary>

Dictionary unpacking into a new dictionary literal (`{**a, **b}`) is one clean way to merge two dicts where the second one's keys win on conflicts; `dict.update()` is another way to express the same idea. Either is fine here.

</details>

## Canonical skill list (starter data, not an exercise)

Save this as `data/canonical_skills.json`. This is a starting list, not something you're expected to hand-write yourself; extend it later as you notice real gaps from the "unmatched skill" logs.

```json
[
  "JavaScript", "TypeScript", "Python", "Java", "C", "C++", "C#", "Go", "Rust",
  "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Dart", "R", "MATLAB", "Perl",
  "Shell scripting", "Bash", "PowerShell", "SQL", "HTML", "CSS", "Sass", "Less",

  "React", "React Native", "Vue.js", "Angular", "Svelte", "Next.js", "Nuxt.js",
  "Remix", "jQuery", "Redux", "Zustand", "Tailwind CSS", "Bootstrap",
  "Material UI", "Styled Components", "Webpack", "Vite", "Babel",

  "Node.js", "Express.js", "NestJS", "Django", "Flask", "FastAPI", "Ruby on Rails",
  "Spring Boot", "Spring Framework", "ASP.NET", ".NET Core", "Laravel", "Symfony",
  "GraphQL", "REST API design", "gRPC", "WebSockets", "Socket.io",

  "PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server", "Oracle Database",
  "MongoDB", "Redis", "Cassandra", "DynamoDB", "Elasticsearch", "Firebase",
  "Supabase", "Neo4j", "InfluxDB", "Database design", "Database normalization",
  "ORM tools", "Prisma", "SQLAlchemy", "TypeORM",

  "AWS", "Amazon EC2", "Amazon S3", "AWS Lambda", "Google Cloud Platform",
  "Microsoft Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
  "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Nginx",
  "Apache HTTP Server", "Linux system administration", "Serverless architecture",
  "Microservices architecture", "Fly.io", "Render", "Heroku", "Vercel", "Netlify",

  "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Trello",
  "Agile methodology", "Scrum", "Kanban", "Test-driven development",
  "Pair programming", "Code review", "Technical documentation writing",

  "Unit testing", "Integration testing", "End-to-end testing", "pytest",
  "Jest", "Mocha", "Cypress", "Playwright", "Selenium", "JUnit",

  "Machine learning", "Deep learning", "Natural language processing",
  "Computer vision", "PyTorch", "TensorFlow", "scikit-learn", "pandas",
  "NumPy", "Data analysis", "Data visualization", "Data engineering",
  "ETL pipelines", "Apache Spark", "Apache Kafka", "Airflow",
  "Large language models", "Prompt engineering", "Vector databases",
  "Embeddings and semantic search",

  "iOS development", "Android development", "Flutter", "SwiftUI",
  "Mobile app development", "Cross-platform development",

  "UI design", "UX design", "Figma", "Sketch", "Adobe XD",
  "Wireframing", "User research", "Accessibility (a11y)", "Responsive design",

  "Project management", "Product management", "Technical leadership",
  "Team leadership", "Mentoring", "Cross-functional collaboration",
  "Stakeholder communication", "Public speaking", "Technical writing",

  "System design", "Software architecture", "Object-oriented design",
  "Functional programming", "Design patterns", "Algorithms and data structures",
  "Performance optimization", "Debugging", "Security best practices",
  "OAuth and authentication", "API security", "Load balancing",
  "Distributed systems", "Message queues", "Caching strategies",

  "English proficiency", "Japanese proficiency", "Bilingual communication",
  "Cross-cultural communication", "Remote work collaboration"
]
```

## Common pitfalls

1. **Re-embedding the canonical list on every request.** This is the exact mistake Chapter 6 warned about in a different shape: if the canonical list gets embedded inside the request handler instead of once at import, you're paying a full embedding pass (local model latency, or real API cost on `openai`) on every single call, for data that never changes.
2. **A nearest-neighbor lookup that never says "I don't know."** Always returning the best-available match, with no threshold, quietly mislabels real gaps in your canonical list as confident matches. Look back at the "spot the bug" example if this one bites you.
3. **Assuming query/passage prefixing works the same way here as in `/match`.** `/match` is asymmetric (a requirement searches over bullets); canonicalizing a skill mention against a canonical name is symmetric (neither side is "searching" the other). Don't carry Chapter 2's asymmetric framing over here without checking whether it actually applies.
4. **Losing the mapping between a mention's position in the request list and its row in a batched similarity grid**, the same ordering discipline Chapter 6 required, now inside `np.argmax(..., axis=1)` instead of a Python loop.

## Explain it back

In a few sentences, no code: why is it fine for the canonical skill embeddings to live only in memory, recomputed once at process start, instead of persisted in the database and cached the way Chapter 6's job and CV embeddings are? What's actually different about this data that makes Chapter 6's whole caching machinery unnecessary here?

## Further reading

- Python `json` docs: <https://docs.python.org/3/library/json.html>
- `numpy.argmax` docs (including the `axis` parameter): <https://numpy.org/doc/stable/reference/generated/numpy.argmax.html>
- A short conceptual read on entity resolution / master data management, the same "many mentions, one real thing" problem at enterprise scale: <https://en.wikipedia.org/wiki/Record_linkage>
- Sentence-Transformers' notes on symmetric vs. asymmetric semantic search, which is the general version of this chapter's prefix contrast with Chapter 2: <https://www.sbert.net/examples/applications/semantic-search/README.html>

## Checkpoint

Before moving to Chapter 8, you should have:
- [ ] `data/canonical_skills.json` committed, holding the starter canonical skill list
- [ ] Canonical skills loaded and embedded exactly once at process start, held in memory, not recomputed per request
- [ ] A working single-mention nearest-neighbor function (Exercise 1)
- [ ] `POST /canonicalize-skills` returning a canonical match and score per input skill, or no match when nothing clears your threshold
- [ ] A threshold that's configurable via an environment variable, chosen by actually testing known-good and known-bad pairs, not guessed
- [ ] Rejected (unmatched) skill mentions logged, not silently dropped
- [ ] Verified in `/docs` with clear matches, paraphrased/multilingual matches, and at least one deliberately unmatched skill
- [ ] (if completed) Exercise 3's batched similarity grid replacing the per-mention embedding loop
- [ ] (if completed) Exercise 5's config-merging function
- [ ] Your answer to Exercise 4's retrieval recall, in `guide/answers.md`
- [ ] Written answer to the "Explain it back" prompt, in `guide/answers.md`
- [ ] Code committed to your repo
