import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers import get_provider
from src.db import insert_embedding, get_embedding

import numpy as np
import uuid

provider = get_provider()
sentences = ["I am a web developer"]

vectors = provider.embed(sentences)
job_id = insert_embedding("job_embeddings", str(uuid.uuid4()), sentences[0], vectors[0], provider.name)
content, vector = get_embedding("job_embeddings", job_id)

print(f"Embedding content: {content}, embedding norm: {np.linalg.norm(vector)}")
