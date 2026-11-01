"""Local PDF policy retrieval: extraction, chunking, embedding, and search.

Store policy documents are a PDF (locally stored). This module extracts their
text with `pypdf`, splits it into deterministic overlapping chunks, embeds
those chunks with a local embedding model served by Ollama, and persists them
in a local ChromaDB collection for retrieval. Nothing here ever sends PDF
content or policy text to a hosted service: embeddings are computed by a
local Ollama server, and the vector store is a local, on-disk ChromaDB
collection.

`retrieve()` never fabricates an answer: when no indexed chunk is close
enough to the query (per `relevance_distance_threshold`), it returns an
empty list so callers can fall back to an honest "I don't know" rather than
inventing policy content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
import pypdf

from store_chat.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_POLICY_CHUNK_OVERLAP,
    DEFAULT_POLICY_CHUNK_SIZE,
    DEFAULT_POLICY_RELEVANCE_DISTANCE_THRESHOLD,
)


class PolicyIndexError(Exception):
    """Raised when a policy PDF cannot be read or contains no usable text."""


@dataclass(frozen=True)
class PolicyChunk:
    """One chunk of extracted policy text, with the source document name."""

    text: str
    source: str


class LocalEmbeddingClient(Protocol):
    """Produces embedding vectors for text using a local embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...


class LocalVectorStore(Protocol):
    """Persists and queries embedded policy chunks."""

    def upsert(self, chunks: list[PolicyChunk], embeddings: list[list[float]]) -> None:
        """Add or replace `chunks` in the store, keyed by their embeddings."""
        ...

    def query(self, embedding: list[float], limit: int) -> list[tuple[PolicyChunk, float]]:
        """Return up to `limit` (chunk, distance) pairs nearest to `embedding`.

        Distance is a non-negative dissimilarity score; smaller means more
        relevant. Results are ordered from most to least relevant.
        """
        ...


class OllamaEmbeddingClient:
    """`LocalEmbeddingClient` backed by a local Ollama server.

    Never contacts a hosted embedding service: `base_url` must point at a
    local Ollama instance (the default is `http://localhost:11434`).
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ) -> None:
        import ollama

        self._model = model
        self._client = ollama.Client(host=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self._model, input=texts)
        return [list(embedding) for embedding in response.embeddings]


class ChromaVectorStore:
    """`LocalVectorStore` backed by a persistent, on-disk ChromaDB collection.

    Data is written under `path` and is readable by any later
    `ChromaVectorStore` constructed against the same path and collection
    name, including in a different process.
    """

    def __init__(self, path: Path, collection_name: str = "policies") -> None:
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        # Cosine distance suits normalized text embeddings; it is also what
        # the relevance threshold in config.py is calibrated against.
        self._collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[PolicyChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        existing_count = self._collection.count()
        ids = [f"chunk-{existing_count + i}" for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source} for chunk in chunks],
        )

    def query(self, embedding: list[float], limit: int) -> list[tuple[PolicyChunk, float]]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=limit)
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            (PolicyChunk(text=document, source=metadata["source"]), distance)
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]


def _extract_text(pdf_path: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises several distinct exception types
        raise PolicyIndexError(f"unreadable PDF '{pdf_path}': {exc}") from exc
    return "\n".join(pages_text)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into overlapping fixed-size character windows.

    Whitespace is first collapsed so a chunk boundary never depends on
    incidental line breaks from PDF extraction. Overlap preserves context
    that would otherwise be lost at a chunk boundary.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


class PolicyIndex:
    """Indexes a policy PDF and answers retrieval queries over it."""

    def __init__(
        self,
        embedding_client: LocalEmbeddingClient,
        vector_store: LocalVectorStore,
        chunk_size: int = DEFAULT_POLICY_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_POLICY_CHUNK_OVERLAP,
        relevance_distance_threshold: float = DEFAULT_POLICY_RELEVANCE_DISTANCE_THRESHOLD,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._relevance_distance_threshold = relevance_distance_threshold

    def index_pdf(self, pdf_path: Path) -> int:
        """Extract, chunk, embed, and persist `pdf_path`. Returns the chunk count.

        Raises `PolicyIndexError` if the PDF cannot be read, or if it
        contains no extractable text (e.g. a blank page).
        """
        text = _extract_text(pdf_path)
        if not text.strip():
            raise PolicyIndexError(
                f"PDF '{pdf_path}' has no extractable text (it may be blank or image-only)"
            )

        chunk_texts = _chunk_text(text, self._chunk_size, self._chunk_overlap)
        source = pdf_path.name
        chunks = [PolicyChunk(text=chunk_text, source=source) for chunk_text in chunk_texts]

        embeddings = self._embedding_client.embed(chunk_texts)
        self._vector_store.upsert(chunks, embeddings)
        return len(chunks)

    def retrieve(self, query: str, limit: int = 4) -> list[PolicyChunk]:
        """Return up to `limit` relevant policy chunks for `query`.

        Returns an empty list when no indexed chunk is close enough to the
        query to be considered relevant (per `relevance_distance_threshold`),
        rather than returning a plausible-looking but unrelated chunk.
        """
        [embedding] = self._embedding_client.embed([query])
        results = self._vector_store.query(embedding, limit)
        return [
            chunk
            for chunk, distance in results
            if distance <= self._relevance_distance_threshold
        ]
