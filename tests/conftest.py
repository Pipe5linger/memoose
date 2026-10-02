import pytest


@pytest.fixture(autouse=True)
def _decider_off(monkeypatch):
    """The decision model is on by default; tests opt in, so a fetched model never changes a result."""
    monkeypatch.setenv("MEMOOSE_DECIDER", "off")

from memoose.store.embeddings import HashEmbedder
from memoose.engine import Engine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMOOSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMOOSE_PROJECT_DIR", str(tmp_path / "proj"))
    eng = Engine(embedder=HashEmbedder())
    yield eng
    eng.close()


@pytest.fixture
def ds(engine):
    return engine.dataset("test")
