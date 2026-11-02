"""Command-line interface for one-off local setup tasks: data import and
policy indexing.

This is a thin wrapper around the same services the running app uses
(`store_chat.importer.import_csv_directory` and
`store_chat.policy_rag.PolicyIndex.index_pdf`): it only parses arguments,
builds the real local dependencies (SQLite connection, Ollama embedding
client, Chroma vector store) from `Settings`, calls the service, and
translates the result (or a raised exception) into a clean English message
and an exit code. No import/indexing business rules are duplicated here.

Usage:
    python -m store_chat.cli import-data --source data/source
    python -m store_chat.cli index-policies --pdf data/source/policies.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from store_chat.config import load_settings
from store_chat.db import get_connection
from store_chat.importer import CsvImportError, import_csv_directory
from store_chat.policy_rag import (
    ChromaVectorStore,
    OllamaEmbeddingClient,
    PolicyIndex,
    PolicyIndexError,
)


def _run_import_data(source: Path) -> int:
    if not source.is_dir():
        print(
            f"Error: source directory '{source}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        return 1

    settings = load_settings()
    connection = get_connection(settings.database_path)
    try:
        report = import_csv_directory(connection, source)
    except CsvImportError as exc:
        print("Error: failed to import CSV data from "
              f"'{source}':", file=sys.stderr)
        for error in exc.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    print(f"Imported data from '{source}' into '{settings.database_path}':")
    for table, count in report.imported_rows.items():
        print(f"  - {table}: {count} row(s)")
    return 0


def _run_index_policies(pdf: Path) -> int:
    if not pdf.is_file():
        print(
            f"Error: policy PDF '{pdf}' does not exist or is not a file.",
            file=sys.stderr,
        )
        return 1

    settings = load_settings()
    embedding_client = OllamaEmbeddingClient(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    vector_store = ChromaVectorStore(path=settings.vector_store_path)
    policy_index = PolicyIndex(
        embedding_client=embedding_client,
        vector_store=vector_store,
        chunk_size=settings.policy_chunk_size,
        chunk_overlap=settings.policy_chunk_overlap,
        relevance_distance_threshold=settings.policy_relevance_distance_threshold,
    )

    try:
        chunk_count = policy_index.index_pdf(pdf)
    except PolicyIndexError as exc:
        print(f"Error: failed to index policy PDF '{pdf}': {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {chunk_count} chunk(s) from '{pdf}' into "
        f"'{settings.vector_store_path}'."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m store_chat.cli",
        description="Store Chat local setup commands: import business data and index store policies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_data_parser = subparsers.add_parser(
        "import-data", help="Import the business CSVs from a source directory."
    )
    import_data_parser.add_argument(
        "--source",
        required=True,
        help="Directory containing the source CSVs (e.g. data/source).",
    )

    index_policies_parser = subparsers.add_parser(
        "index-policies", help="Index a store policies PDF for retrieval."
    )
    index_policies_parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the store policies PDF (e.g. data/source/policies.pdf).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` and run the requested subcommand.

    Returns a non-zero exit code with a clear English error message on
    stderr for invalid or missing inputs; returns 0 and prints a short
    success summary to stdout otherwise.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "import-data":
        return _run_import_data(Path(args.source))
    if args.command == "index-policies":
        return _run_index_policies(Path(args.pdf))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
