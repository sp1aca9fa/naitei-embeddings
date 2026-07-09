import sys
import uuid
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers import get_provider
from src.db import text_to_hash, get_or_create_embedding

from .helpers.db_helpers import check_row_count

def test_same_provider(cleanup_hashes):
    provider = get_provider()
    count_before = check_row_count("job_embeddings")
    sentence = str(uuid.uuid4())
    vec1 = get_or_create_embedding("job_embeddings", sentence, provider)
    cleanup_hashes.append(("job_embeddings", text_to_hash(sentence), provider.name))
    vec2 = get_or_create_embedding("job_embeddings", sentence, provider)
    cleanup_hashes.append(("job_embeddings", text_to_hash(sentence), provider.name))
    count_after = check_row_count("job_embeddings")
    assert count_before + 1 == count_after
    assert np.array_equal(vec1, vec2)


def test_different_providers(cleanup_hashes):
    provider = get_provider()
    provider2 = get_provider("mock")
    count_before = check_row_count("job_embeddings")
    sentence = str(uuid.uuid4())
    vec1 = get_or_create_embedding("job_embeddings", sentence, provider)
    cleanup_hashes.append(("job_embeddings", text_to_hash(sentence), provider.name))
    vec2 = get_or_create_embedding("job_embeddings", sentence, provider2)
    cleanup_hashes.append(("job_embeddings", text_to_hash(sentence), provider2.name))
    count_after = check_row_count("job_embeddings")
    assert count_before + 2 == count_after
    assert not np.array_equal(vec1, vec2)
