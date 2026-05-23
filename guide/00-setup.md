# Chapter 0: Project Setup

## What we're building in this chapter

A clean Python 3.12 project with a virtual environment, the starter folder layout, and a minimal FastAPI app that answers `GET /health` with `OK`. By the end you will have committed the scaffolding to git and run the server locally with `uvicorn`.

## Why this matters

Every Python project starts here, and getting this right saves hours later. Three things in particular set the tone for the whole service:

1. **Isolation.** Python's global package store is a swamp. A virtual environment gives this project its own pinned set of dependencies, so nothing you install for `naitei-embeddings` leaks into other Python work on your machine, and nothing on your machine silently affects this project.

2. **A health endpoint.** Long-running services need a single cheap URL that says "I am alive". Fly.io will poll it. Naitei will poll it before sending real traffic. Building it first means deployment in Chapter 11 has one less moving part.

3. **The shape of the repo.** Putting `src/`, `tests/`, `scripts/`, and `guide/` in place now means every later chapter can drop files into the right home without you stopping to refactor.

You are also doing your first commit, which forces the `.gitignore` to be correct before any junk gets tracked. Once a virtualenv folder or a `.env` file is in git history, getting it out is annoying.

## Concepts

### Virtual environments

A virtual environment is a folder containing its own copy of the Python interpreter and its own `site-packages` directory where `pip install` writes packages. When the venv is "activated", your shell's `python` and `pip` point at that folder instead of the system Python.

You came from CS50P where everything was installed globally (or in the CS50 codespace, which is itself a kind of sandbox). In real Python work, the rule is one venv per project. The TypeScript analogue is `node_modules`: a per-project folder of dependencies that nothing else touches. The difference is that Node finds `node_modules` automatically, while Python needs you to activate the venv first so the right binaries are on your `PATH`.

The Python 3 standard library ships a venv module, so you do not need to install anything to create one.

### ASGI and why FastAPI exists

Flask, which you used in CS50P, is a WSGI framework. WSGI is a synchronous specification: one request, one thread, blocking I/O. That model is fine for CPU-bound work or small apps, but it means a request that is waiting on a slow database or a slow embedding model holds a worker thread the whole time.

FastAPI is an ASGI framework. ASGI is the async successor to WSGI. A FastAPI route can be declared `async def`, and while it is awaiting something slow, the event loop is free to handle other requests. This matters for us because embedding calls (especially against OpenAI) are network-bound and we want concurrency without spinning up dozens of threads.

If you have written an Express server in Node, ASGI will feel familiar. Express handlers can `await` an async database call and Node's event loop handles the rest. FastAPI gives Python the same shape.

The two other things FastAPI does that Flask does not, and that you will feel almost immediately:

- **Pydantic-based request and response models.** You declare your input and output shapes as Python classes, and FastAPI both validates incoming JSON and serializes outgoing JSON for you. Think Zod, but built in.
- **Automatic OpenAPI docs.** FastAPI generates a Swagger UI at `/docs` from your route signatures. You do not write the spec; it reads your code.

You will not use either of these in Chapter 0 because `/health` is trivial. Both arrive in earnest in Chapter 5.

### The ASGI server (uvicorn)

A FastAPI app is just an object. Something else has to actually listen on a TCP port, accept HTTP requests, and call the app. That something is an ASGI server, and the default choice is `uvicorn`. The mental model:

```
client  ->  uvicorn (HTTP server)  ->  FastAPI app  ->  your route function
```

This is the same split as `gunicorn` plus Flask, or `node` plus Express. In development you run uvicorn directly with `--reload` so the server restarts when you edit a file. In production (Chapter 11) you will run uvicorn behind a process supervisor inside a Docker container.

### The 12-factor `.env` pattern

Production config (API keys, database URLs, the shared secret) does not belong in source code. The convention is:

- A `.env` file in the project root holds the real values. It is gitignored.
- A `.env.example` file lists the same keys with placeholder or empty values. It is committed, so anyone cloning the repo knows what they need to set.
- The application reads values from environment variables at startup.

You will not load any env vars in Chapter 0, but you will set up the `.env.example` skeleton so the pattern is in place from day one.

## The tools we're using

### Python 3.12

- What it is: the Python interpreter, current stable major version when this project was scoped.
- What it does for us: runs everything we write.
- Install: use `pyenv`, `asdf`, or whatever your system prefers. Check with `python3.12 --version`. The shipped `python3` on your machine may be 3.10 or 3.11; that is fine as long as `python3.12` exists.
- Docs: <https://docs.python.org/3.12/>

### `venv` (standard library)

- What it is: the built-in virtual environment tool.
- What it does for us: creates an isolated Python environment in a folder named `.venv`.
- Install: nothing to install, it ships with Python.
- Docs: <https://docs.python.org/3.12/library/venv.html>
- Key commands:
  - `python3.12 -m venv .venv` creates the environment.
  - `source .venv/bin/activate` activates it in the current shell. Your prompt usually gains a `(.venv)` prefix.
  - `deactivate` exits the venv.

### `pip`

- What it is: Python's package installer.
- What it does for us: reads `requirements.txt` and downloads packages into the active venv.
- Install: ships with Python.
- Docs: <https://pip.pypa.io/en/stable/>
- Key commands:
  - `pip install -r requirements.txt` installs every pinned dependency.
  - `pip install fastapi` installs one package and adds nothing to `requirements.txt` automatically. You edit `requirements.txt` by hand or with `pip freeze`.

### FastAPI

- What it is: a modern async Python web framework built on Starlette and Pydantic.
- What it does for us: defines our HTTP routes and handles request/response serialization.
- Install: `pip install fastapi`
- Docs: <https://fastapi.tiangolo.com/>
- Key pieces we use in this chapter:
  - `FastAPI()` constructs the application object.
  - `@app.get("/path")` decorates a function as a GET route. The same decorator pattern exists for `post`, `put`, `delete`.

### `uvicorn`

- What it is: an ASGI server.
- What it does for us: runs the FastAPI app on a local port.
- Install: `pip install "uvicorn[standard]"`. The `[standard]` extra pulls in performance-related optional dependencies (`uvloop`, `httptools`, etc.). You want them.
- Docs: <https://www.uvicorn.org/>
- Key command:
  - `uvicorn src.main:app --reload` runs the `app` object from the `src/main.py` module, restarting on file changes.

## How it fits together

By the end of this chapter your directory will look like this:

```
naitei-embeddings/
├── PROJECT.md
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── guide/
│   └── 00-setup.md
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   └── __init__.py
└── scripts/
```

The data flow is dead simple right now:

```
curl http://localhost:8000/health
        |
        v
   uvicorn (port 8000)
        |
        v
   FastAPI app (src/main.py)
        |
        v
   health() returns {"status": "ok"}
        |
        v
   uvicorn serializes to JSON and sends 200
```

The `__init__.py` files (empty) tell Python that `src/` and `tests/` are packages, which lets `uvicorn` resolve `src.main:app` and lets pytest later discover tests by package path.

## Code examples

### Example 1: The smallest possible FastAPI app

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root() -> dict[str, str]:
    """Return a friendly greeting."""
    return {"message": "hello"}
```

Run with `uvicorn example:app --reload`, then:

```
$ curl http://localhost:8000/
{"message":"hello"}
```

The return value is a plain Python dict. FastAPI serializes it to JSON automatically and sets `Content-Type: application/json`. No template engine, no `jsonify()` wrapper as in Flask.

### Example 2: Multiple routes on one app

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    """Report service liveness."""
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    """Report the running version."""
    return {"version": "0.1.0"}
```

```
$ curl http://localhost:8000/health
{"status":"ok"}
$ curl http://localhost:8000/version
{"version":"0.1.0"}
```

Each decorated function becomes a route. The function name does not have to match the path; only the decorator argument does.

### Example 3: Returning a plain string

If you want the response body to be the literal string `OK` rather than JSON, you return a string but FastAPI will still wrap it in quotes as JSON:

```python
@app.get("/health")
def health() -> str:
    return "OK"
```

```
$ curl http://localhost:8000/health
"OK"
```

This is fine for a health endpoint. Many real services return a JSON object so they can add fields later without breaking clients. Either is acceptable for this chapter; pick one.

### Example 4: What a typical `requirements.txt` looks like

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
```

Pinning exact versions (`==`) keeps the build reproducible. You will add more lines in later chapters. Use whatever the current stable versions are when you install; the exact numbers above are illustrative.

## Your task

Create the project scaffolding and a running FastAPI service with a `/health` endpoint. Specifically:

1. **Initialize git.** From inside `naitei-embeddings/`, run `git init` if it is not already a repo.

2. **Create the virtual environment.**
   - Run `python3.12 -m venv .venv`.
   - Activate it: `source .venv/bin/activate`.
   - Confirm with `python --version` (should print 3.12.x) and `which python` (should point inside `.venv/`).

3. **Write `.gitignore`.** At minimum it must exclude `.venv/`, `__pycache__/`, `.env`, and any IDE folders you use. If you forget this, the next step will start tracking the venv and you will have to clean it up.

4. **Write `requirements.txt`** with `fastapi` and `uvicorn[standard]`. Pin to specific versions you choose (look up the current versions on PyPI).

5. **Install dependencies:** `pip install -r requirements.txt`.

6. **Create the source layout.**
   - `src/__init__.py` (empty file).
   - `src/main.py` with a FastAPI app instance and a single `GET /health` route. The handler returns either `{"status": "ok"}` or the string `"OK"`. Add a docstring.
   - `tests/__init__.py` (empty file). No tests yet; you are just claiming the folder.
   - `scripts/` directory exists but can be empty for now (git will not track an empty folder; ignore this for the commit).

7. **Write `.env.example`.** Leave it empty except for a comment line explaining its purpose. Real keys arrive in later chapters.

8. **Write a minimal `README.md`** with the project name, a one-sentence description, and a short "Local development" section showing how to activate the venv and run the server. Aim for under 30 lines.

9. **Run the server.** From the project root with the venv active:

   ```
   uvicorn src.main:app --reload
   ```

   Then in another terminal:

   ```
   curl http://localhost:8000/health
   ```

   You should see your `OK` or `{"status":"ok"}` response. Also open <http://localhost:8000/docs> in a browser and confirm FastAPI's automatic Swagger UI loads and lists your route.

10. **Commit.** `git add` the tracked files (not `.venv/`, not `.env`), then commit with a message like `chore: initial project scaffolding`.

## Common pitfalls

1. **The venv is not active when you `pip install`.** Symptoms: packages install to your global Python, `pip list` inside the venv shows nothing useful, `uvicorn` is "not found" after install. Fix: re-run `source .venv/bin/activate`, confirm the `(.venv)` prefix appears in your prompt, then reinstall.

2. **You forgot to create `src/__init__.py`.** Symptoms: `uvicorn src.main:app` fails with `ModuleNotFoundError: No module named 'src'`. Fix: create the empty file. The presence of `__init__.py` is what makes `src` a Python package.

3. **You ran `uvicorn` from inside `src/`.** Symptoms: same `ModuleNotFoundError`. Fix: always run uvicorn from the project root so Python's import path includes the parent of `src/`.

4. **`.venv/` got committed because `.gitignore` was missing or wrong.** Symptoms: `git status` lists hundreds of files inside `.venv/`. Fix: add `.venv/` to `.gitignore`, then `git rm -r --cached .venv` to untrack what was already added, then commit.

5. **Port 8000 is already in use.** Symptoms: uvicorn exits with `[Errno 98] Address already in use`. Fix: either find and stop the other process, or run uvicorn on a different port with `--port 8001`.

6. **You installed `uvicorn` without the `[standard]` extra and `--reload` does nothing useful.** Symptoms: reload sometimes does not detect changes, or warnings about missing `watchfiles`. Fix: `pip install "uvicorn[standard]"` and re-pin in `requirements.txt`. The quotes around `uvicorn[standard]` matter in zsh because brackets are glob characters.

## Stuck? Hints (click to expand)

<details>
<summary>Hint 1 — Conceptual nudge</summary>

If `uvicorn src.main:app` fails to find your app, walk through the dotted path mentally. The first part (`src.main`) is the module path Python uses to import. The part after the colon (`app`) is the name of the variable in that module. So the question is: from the directory you are running `uvicorn` in, can Python import `src.main`, and does that module expose a top-level variable called `app`?

</details>

<details>
<summary>Hint 2 — Approach and pseudocode</summary>

The shape of `src/main.py` is:

```
import the FastAPI class
create an app instance assigned to a module-level variable called `app`
define a function for /health, decorated with @app.get("/health")
the function returns either a string or a dict
```

The shape of your shell session is:

```
cd into project root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

If a step fails, do not skip ahead. The next step will fail for a confusing reason.

</details>

<details>
<summary>Hint 3 — Code skeleton</summary>

`src/main.py`:

```python
from fastapi import FastAPI

app = FastAPI(title="naitei-embeddings")


@app.get("/health")
def health() -> dict[str, str]:
    """Report service liveness for uptime checks."""
    return {"status": "ok"}
```

`.gitignore` starter:

```
.venv/
__pycache__/
*.pyc
.env
.DS_Store
.vscode/
.idea/
```

`.env.example` starter:

```
# Copy this file to .env and fill in real values.
# Real secrets go in .env, which is gitignored.
```

`requirements.txt` starter (use current versions from PyPI):

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
```

</details>

## Further reading

- FastAPI's official first-steps tutorial: <https://fastapi.tiangolo.com/tutorial/first-steps/>
- Python `venv` documentation: <https://docs.python.org/3.12/library/venv.html>
- The Twelve-Factor App, factor III "Config": <https://12factor.net/config>
- ASGI vs WSGI, short overview: <https://asgi.readthedocs.io/en/latest/introduction.html>

## Checkpoint

Before moving to Chapter 1, you should have:

- [ ] A `.venv/` virtual environment using Python 3.12, activated successfully
- [ ] `requirements.txt` pinning `fastapi` and `uvicorn[standard]`
- [ ] `.gitignore` excluding `.venv/`, `.env`, `__pycache__/`
- [ ] `.env.example` committed (empty or with a comment)
- [ ] `src/__init__.py` and `src/main.py` with a working `/health` route
- [ ] `tests/__init__.py` placeholder
- [ ] A short `README.md`
- [ ] `uvicorn src.main:app --reload` serving 200 on `GET /health` locally
- [ ] FastAPI's auto-docs visible at `http://localhost:8000/docs`
- [ ] First commit made, with no venv files or `.env` tracked
