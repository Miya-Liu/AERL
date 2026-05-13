import pytest

from aerl.settings import join_upstream_subpath, load_settings, normalize_upstream_base


def test_missing_upstream_raises(monkeypatch):
    monkeypatch.delenv("UPSTREAM_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AERL_DATA_DIR", "/tmp/aerl")
    with pytest.raises(ValueError, match="UPSTREAM_OPENAI_BASE_URL"):
        load_settings()


def test_missing_data_dir_raises(monkeypatch):
    monkeypatch.setenv("UPSTREAM_OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.delenv("AERL_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="AERL_DATA_DIR"):
        load_settings()


def test_normalize_and_join_no_double_slash(monkeypatch):
    monkeypatch.setenv("UPSTREAM_OPENAI_BASE_URL", "https://x/v1/")
    monkeypatch.setenv("AERL_DATA_DIR", "/tmp/aerl")
    s = load_settings()
    assert s.upstream_openai_base_url == "https://x/v1"
    assert join_upstream_subpath(s.upstream_openai_base_url, "chat/completions") == (
        "https://x/v1/chat/completions"
    )


def test_normalize_upstream_base_invalid():
    with pytest.raises(ValueError, match="scheme and host"):
        normalize_upstream_base("not-a-url")
