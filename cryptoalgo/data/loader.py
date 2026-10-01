"""CSV import/export and a convenience loader used by scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .integrity import clean_frame
from .models import OHLCV


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "open_time"
    out.to_csv(path)


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV with columns open_time (ISO or epoch ms), open, high, low, close, volume."""
    df = pd.read_csv(path)
    col = "open_time" if "open_time" in df.columns else df.columns[0]
    t = df[col]
    if pd.api.types.is_numeric_dtype(t):
        idx = pd.to_datetime(t, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    out = df.drop(columns=[col])[OHLCV].astype(float)
    out.index = pd.DatetimeIndex(idx)
    return clean_frame(out)


def load_dir(directory: str | Path, symbols, timeframe: str) -> dict[str, pd.DataFrame]:
    d = Path(directory)
    return {s: load_csv(d / f"{s}_{timeframe}.csv") for s in symbols}


def load_data(source: str, symbols, timeframe: str) -> tuple[dict[str, pd.DataFrame], str]:
    """``source`` = a directory of ``<SYMBOL>_<tf>.csv`` files, or ``synthetic:<world>:<seed>:<days>[:<edge_scale>]``.

    Returns (data, provenance). Provenance is "REAL" for CSV directories and
    "SYNTHETIC (...)" otherwise, so reports can never confuse the two.
    """
    if source.startswith("synthetic:"):
        from .synthetic import generate
        parts = source.split(":")
        _, world, seed, days = parts[:4]
        scale = float(parts[4]) if len(parts) > 4 else 1.0
        data, _ = generate(tuple(symbols), days=int(days), timeframe=timeframe, seed=int(seed), world=world,
                           edge_scale=scale)
        extra = f", edge_scale={scale}" if scale != 1.0 else ""
        return data, f"SYNTHETIC ({world}, seed={seed}, {days}d{extra}) - NOT REAL MARKET DATA"
    return load_dir(source, symbols, timeframe), f"REAL/CSV ({source})"
