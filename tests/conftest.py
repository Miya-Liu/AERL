import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Default env for AERL integration tests; tests that need missing vars use monkeypatch."""
    os.environ.setdefault("UPSTREAM_OPENAI_BASE_URL", "https://upstream.test/v1")
    if "AERL_DATA_DIR" not in os.environ:
        d = tempfile.mkdtemp(prefix="aerl-data-")
        config._aerl_session_data_dir = d  # type: ignore[attr-defined]
        os.environ["AERL_DATA_DIR"] = d


def pytest_unconfigure(config: pytest.Config) -> None:
    d = getattr(config, "_aerl_session_data_dir", None)
    if d and Path(d).exists():
        shutil.rmtree(d, ignore_errors=True)
    if d and str(d).startswith(tempfile.gettempdir()) and "aerl-data-" in str(d):
        os.environ.pop("AERL_DATA_DIR", None)


def _can_bind_ipv4_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


def _can_bind_ipv6_loopback() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def _has_ipv4_route() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return bool(ip) and not ip.startswith("127.")
    except OSError:
        return False


def _has_ipv6_route() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            s.connect(("2001:4860:4860::8888", 80))
            ip = s.getsockname()[0]
        return bool(ip) and ip != "::1"
    except OSError:
        return False


@pytest.fixture(scope="session")
def ip_stack() -> dict[str, bool]:
    ipv4_loopback = _can_bind_ipv4_loopback()
    ipv6_loopback = _can_bind_ipv6_loopback()
    ipv4_route = _has_ipv4_route()
    ipv6_route = _has_ipv6_route()
    return {
        "ipv4_loopback": ipv4_loopback,
        "ipv6_loopback": ipv6_loopback,
        "ipv4_route": ipv4_route,
        "ipv6_route": ipv6_route,
    }
