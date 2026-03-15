import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


def make_doc(text: str, metadata: dict = {}) -> Document:
    return Document(page_content=text, metadata=metadata)


# --- Cleaner ---

def test_clean_text_removes_extra_spaces():
    from src.ingestion.cleaner import clean_text
    assert clean_text("hello   world") == "hello world"


def test_clean_text_strips():
    from src.ingestion.cleaner import clean_text
    assert clean_text("  hello  ") == "hello"


# --- Splitter ---

def test_split_documents_returns_chunks():
    from src.chunking.splitter import split_documents
    long_text = "Mot " * 300
    docs = [make_doc(long_text)]
    chunks = split_documents(docs)
    assert len(chunks) > 1


def test_split_preserves_metadata():
    from src.chunking.splitter import split_documents
    docs = [make_doc("Mot " * 300, metadata={"name": "Alice"})]
    chunks = split_documents(docs)
    assert all(c.metadata.get("name") == "Alice" for c in chunks)


# --- Retriever ---

def test_retrieve_calls_similarity_search():
    from src.retrieval.retriever import retrieve
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [make_doc("CV texte")]
    results = retrieve(mock_vs, "Développeur Python", k=3)
    mock_vs.similarity_search.assert_called_once_with("Développeur Python", k=3)
    assert len(results) == 1
