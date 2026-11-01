"""Tests for local PDF policy extraction, chunking, and persistent retrieval.

Embeddings are produced by a small deterministic, in-memory fake (a hashed
bag-of-words vector) so these tests never touch a real Ollama server. The
vector store under test is the real `ChromaVectorStore`, because the
persistence behavior (surviving a fresh object construction) is part of the
contract being verified.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from store_chat.policy_rag import ChromaVectorStore, PolicyIndex, PolicyIndexError

_EMBEDDING_DIM = 1024


class FakeEmbeddingClient:
    """Deterministic, in-memory stand-in for a local Ollama embedding model.

    Produces a normalized signed-hashed bag-of-words vector for each text
    (the "hashing trick" with a random +1/-1 sign per word, which reduces
    bias from hash collisions compared to plain counting): words are
    tokenized and lowercased, each hashed via stable md5 (not Python's salted
    `hash()`, which is not reproducible across processes) into one of
    `_EMBEDDING_DIM` buckets with a sign also derived from a stable hash, and
    the resulting vector is L2-normalized. Texts sharing vocabulary end up
    with small cosine distance; texts with no shared vocabulary end up
    orthogonal (distance 1.0). This is simple and fully deterministic across
    runs and processes, unlike a real embedding model, while still giving
    tests meaningful, reproducible separation between relevant and
    irrelevant text.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * _EMBEDDING_DIM
        for word in re.findall(r"[a-zA-Z]+", text.lower()):
            word_hash = hashlib.md5(word.encode("utf-8")).hexdigest()
            bucket = int(word_hash, 16) % _EMBEDDING_DIM
            sign_hash = hashlib.md5((word + ":sign").encode("utf-8")).hexdigest()
            sign = 1.0 if int(sign_hash, 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


def make_policy_pdf(path: Path, paragraphs: list[str]) -> Path:
    """Write a small PDF at `path` with one paragraph of text per line, via ReportLab."""
    pdf = canvas.Canvas(str(path))
    y = 750
    for paragraph in paragraphs:
        pdf.drawString(50, y, paragraph)
        y -= 20
    pdf.save()
    return path


def make_blank_pdf(path: Path) -> Path:
    """Write a PDF at `path` with a single page containing no text at all."""
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()
    return path


POLICY_PARAGRAPHS = [
    "Return Policy: Returns are accepted within thirty days of purchase.",
    "A valid receipt and original packaging are required for any refund.",
    "Warranty Policy: All electric guitars include a one year manufacturer warranty.",
    "Shipping Policy: Orders ship within two business days from our local warehouse.",
]


@pytest.fixture
def policy_pdf_path(tmp_path: Path) -> Path:
    return make_policy_pdf(tmp_path / "policies.pdf", POLICY_PARAGRAPHS)


def build_index(vector_store_path: Path, collection_name: str = "policies") -> PolicyIndex:
    return PolicyIndex(
        embedding_client=FakeEmbeddingClient(),
        vector_store=ChromaVectorStore(path=vector_store_path, collection_name=collection_name),
        # Small chunk size relative to production's default (500/50), chosen
        # so this short synthetic document's four distinct policy topics
        # land in separate chunks instead of being merged into one.
        chunk_size=90,
        chunk_overlap=10,
    )


def test_index_pdf_persists_chunks(tmp_path: Path, policy_pdf_path: Path) -> None:
    vector_store_path = tmp_path / "chroma"

    first_index = build_index(vector_store_path)
    chunk_count = first_index.index_pdf(policy_pdf_path)
    assert chunk_count > 0

    # A brand new PolicyIndex (and a brand new ChromaVectorStore instance)
    # pointed at the same persistence path must see the previously indexed
    # chunks without re-indexing, proving persistence survives a fresh
    # construction rather than relying on an in-memory cache.
    second_index = build_index(vector_store_path)
    results = second_index.retrieve("How many days do I have to return an item?")

    assert len(results) > 0
    assert any("thirty days" in chunk.text for chunk in results)
    assert all(chunk.source == policy_pdf_path.name for chunk in results)


def test_index_pdf_rejects_unreadable_pdf(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.pdf"
    corrupted_path.write_bytes(b"this is not a real PDF file, just garbage bytes")

    index = build_index(tmp_path / "chroma")

    with pytest.raises(PolicyIndexError, match="unreadable"):
        index.index_pdf(corrupted_path)


def test_index_pdf_rejects_empty_pdf(tmp_path: Path) -> None:
    blank_pdf_path = make_blank_pdf(tmp_path / "blank.pdf")

    index = build_index(tmp_path / "chroma")

    with pytest.raises(PolicyIndexError, match="no extractable text"):
        index.index_pdf(blank_pdf_path)


def test_retrieve_returns_relevant_chunks_with_source(
    tmp_path: Path, policy_pdf_path: Path
) -> None:
    index = build_index(tmp_path / "chroma")
    index.index_pdf(policy_pdf_path)

    results = index.retrieve("What is the warranty on electric guitars?", limit=4)

    assert len(results) > 0
    assert len(results) <= 4
    assert any("warranty" in chunk.text.lower() for chunk in results)
    assert all(chunk.source == policy_pdf_path.name for chunk in results)


def test_retrieve_returns_empty_for_irrelevant_query(
    tmp_path: Path, policy_pdf_path: Path
) -> None:
    index = build_index(tmp_path / "chroma")
    index.index_pdf(policy_pdf_path)

    # Shares essentially no vocabulary with any indexed policy paragraph, so
    # the fake embedding places it far from every chunk: retrieve must return
    # an empty list rather than fabricating a plausible-looking answer.
    results = index.retrieve("spacecraft turbine orbital mechanics telemetry")

    assert results == []
