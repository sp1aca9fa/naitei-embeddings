import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers import get_provider
from src.db import insert_embedding, search_embedding

import numpy as np
import uuid

provider = get_provider()

sentences = ["Experience with React", "TypeScript の実務経験", "Backend API design", "チームでの開発経験"]

for s in range(len(sentences)):
    insert_embedding("job_embeddings", str(uuid.uuid4()), sentences[s], provider.embed(sentences)[s], provider.name)

query_sentences = ["Frontend developer with React"]
query_vec = provider.embed(sentences)[0]

results = search_embedding("job_embeddings", query_vec, 10)

print(f"Query: {sentences[0]}\n")
for source_text, model_name, similarity in results:
    print(f"{similarity:.3f}  {source_text}  [{model_name}]")
