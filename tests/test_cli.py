"""Tests for the Store Chat command-line interface.

The CLI is a thin wrapper around `store_chat.importer.import_csv_directory`
and `store_chat.policy_rag.PolicyIndex.index_pdf`: these tests only check
that invalid/missing inputs produce a non-zero exit code and a clear English
error message, without duplicating the importer's or indexer's own
validation logic (that is covered by `tests/test_importer.py` and
`tests/test_policy_rag.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from store_chat.cli import main


def test_cli_import_data_reports_missing_source_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_dir = tmp_path / "does-not-exist"

    exit_code = main(["import-data", "--source", str(missing_dir)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "does-not-exist" in captured.err
    assert "source directory" in captured.err.lower()


def test_cli_index_policies_reports_missing_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_pdf = tmp_path / "policies.pdf"

    exit_code = main(["index-policies", "--pdf", str(missing_pdf)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "policies.pdf" in captured.err
    assert "pdf" in captured.err.lower()
