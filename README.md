# Naitei Embeddings

## Description

A Python API running on FastAPI and uvicorn to handle Machine Learning embeddings from Naitei (hosted in a separated repository).

(more details to be added as the app is built)

## Local development

Run the commands below from the project root folder

Create a virtual environment
```
python3.12 -m venv .venv
```

Activate the virtual environment
```
source .venv/bin/activate
```

Install the requirements
```
pip install -r requirements.txt
```

Referring to .env.example, create a ".env" file containing all environment variables with your own variables.

Run the server
```
uvicorn src.main:app --reload
```

## Tests

### Check health

```
curl http://localhost:8000/health
```
