"""Tests for Settings/load_settings .env file loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from store_chat.config import load_settings


def test_load_settings_reads_values_from_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "OLLAMA_CHAT_MODEL=test-model-from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.chat_model == "test-model-from-dotenv"


def test_load_settings_reads_gemini_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    settings = load_settings()

    assert settings.gemini_api_key == "test-key-123"
