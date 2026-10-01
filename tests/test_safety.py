"""Paper-only guarantees (requirement 11)."""
import re
from pathlib import Path

import pytest

from cryptoalgo.config import ConfigError, load_config
from cryptoalgo.safety import SafetyError, assert_paper_only

SRC = Path(__file__).resolve().parent.parent / "cryptoalgo"


def test_mode_must_be_paper():
    with pytest.raises(ConfigError):
        load_config(overrides={"mode": "live"})


def test_assert_paper_only_rejects_credentials_in_env(monkeypatch):
    cfg = load_config()
    assert_paper_only(cfg)
    monkeypatch.setenv("BINANCE_API_KEY", "x")
    with pytest.raises(SafetyError):
        assert_paper_only(cfg)


def test_unknown_config_keys_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"risk": {"leverage_yolo": 100}})
    with pytest.raises(ConfigError):
        load_config(overrides={"api_key": "abc"})


# Source scan: nothing capable of real trading may exist anywhere in the package.
FORBIDDEN = [r"/api/v3/order", r"/fapi/", r"/sapi/", r"X-MBX-APIKEY", r"hmac", r"hashlib\.sha256", r"signature",
             r"withdraw", r"api[_-]?secret", r"api[_-]?key", r"new_order", r"create_order", r"ccxt", r"python-binance"]


def test_no_real_trading_or_signing_code_in_source():
    offenders = []
    for p in SRC.rglob("*"):
        if p.suffix not in (".py", ".html", ".toml"):
            continue
        text = p.read_text()
        for pat in FORBIDDEN:
            for m in re.finditer(pat, text, flags=re.I):
                line = text[:m.start()].count("\n") + 1
                ctx = text.splitlines()[line - 1].strip()
                # the safety module and tests describe the ban in prose; allow only comments/docstrings there
                if p.name == "safety.py":
                    continue
                offenders.append(f"{p.relative_to(SRC.parent)}:{line}: {ctx}")
    assert not offenders, "forbidden real-trading related code:\n" + "\n".join(offenders)


def test_no_exchange_sdk_dependency_declared():
    root = SRC.parent
    deps = (root / "requirements.txt").read_text() + (root / "pyproject.toml").read_text()
    for bad in ("ccxt", "python-binance", "binance-connector", "pybit", "krakenex"):
        assert bad not in deps


def test_http_client_only_issues_get_to_klines():
    text = (SRC / "data" / "binance.py").read_text()
    assert ".post(" not in text and ".put(" not in text and ".delete(" not in text
    assert text.count("KLINE_PATH") >= 2
