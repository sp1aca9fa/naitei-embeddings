import pytest
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import delete_embedding_by_hash

@pytest.fixture
def cleanup_hashes():
    created = []
    yield created
    for table_name, content_hash, model_name in created:
        delete_embedding_by_hash(table_name, content_hash, model_name)
