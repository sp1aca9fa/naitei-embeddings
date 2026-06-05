import pytest
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.providers import get_provider
from src.providers import MockProvider

SENTENCES = [
        "Frontend engineer",
        "React エンジニア",
        "Most people never visited Botsuana"
    ]

def test_mock_is_deterministic():
    mock1 = MockProvider(64)
    mock2 = MockProvider(64)
    assert np.array_equal(mock1.embed(SENTENCES), mock2.embed(SENTENCES))

def test_mock_vectors_are_unit_normalized():
    mock = MockProvider(365)
    embeddings = mock.embed(SENTENCES)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_mock_dimension_is_respect():
    mock = MockProvider(128)
    embeddings = mock.embed(SENTENCES)
    assert embeddings.shape == (3, 128)

def test_mock_different_texts_give_different_vectors():
    mock = MockProvider(252)
    a = mock.embed(["React developer"])
    b = mock.embed(["Vue developer"])
    assert not np.array_equal(a, b)
