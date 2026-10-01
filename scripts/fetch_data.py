#!/usr/bin/env python3
"""Download historical candles from Binance PUBLIC endpoints (no API key) into CSV files.

  python scripts/fetch_data.py --symbols BTCUSDT ETHUSDT --timeframe 1h --start 2020-01-01 --out data/
Needs outbound HTTPS to api.binance.com (or data-api.binance.vision).
"""
import argparse
import sys

import pandas as pd

from cryptoalgo.data.binance import BinancePublicClient, MarketDataError
from cryptoalgo.data.integrity import check_candles
from cryptoalgo.data.loader import save_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="default: now")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    cli = BinancePublicClient()
    end = pd.Timestamp(a.end, tz="UTC") if a.end else pd.Timestamp.now(tz="UTC")
    for s in a.symbols:
        try:
            df = cli.history(s, a.timeframe, pd.Timestamp(a.start, tz="UTC"), end)
        except MarketDataError as e:
            print(f"{s}: download failed: {e}\nCheck that this machine can reach api.binance.com "
                  "(some networks/regions block it).", file=sys.stderr)
            return 1
        rep = check_candles(df, a.timeframe)
        save_csv(df, f"{a.out}/{s}_{a.timeframe}.csv")
        print(f"{s}: {len(df)} bars {df.index[0]} -> {df.index[-1]}  integrity_ok={rep.ok} gaps={len(rep.gaps)} {rep.issues}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
