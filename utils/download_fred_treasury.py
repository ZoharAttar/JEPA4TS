
"""Download US Treasury constant-maturity yields from FRED into a TSLib CSV.

Eight daily tenors (business-day), last column named OT (30Y), same schema as
exchange_rate.csv. No FRED API key. Cite: Diebold & Li (2006); series are FRED
DGS*.

Usage (from repo root):
    python utils/download_fred_treasury.py
    python utils/download_fred_treasury.py --out dataset/treasury_yields/treasury_yields.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# 3M–30Y; inner-join starts ~1982 (when DGS3MO begins). enc_in = 8.
# OT is the 30Y yield (Dataset_Custom requires --target OT).
FRED_IDS = [
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS30",
]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_series(sid: str) -> pd.Series:
    url = FRED_CSV.format(sid=sid)
    raw = pd.read_csv(url, na_values=["."])
    date_col = raw.columns[0]
    val_col = raw.columns[1]
    s = pd.to_numeric(raw[val_col], errors="coerce")
    s.index = pd.to_datetime(raw[date_col])
    s.name = sid
    return s


def build_panel() -> pd.DataFrame:
    parts = [fetch_series(sid) for sid in FRED_IDS]
    df = pd.concat(parts, axis=1).dropna(how="any")
    df = df.sort_index()
    df.index.name = "date"
    # Unique 8 channels: 3M..10Y plus OT = 30Y (do not keep DGS30 twice).
    out = df[FRED_IDS[:-1]].copy()
    out["OT"] = df["DGS30"]
    out = out.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FRED Treasury curve → TSLib CSV")
    parser.add_argument(
        "--out",
        type=str,
        default="dataset/treasury_yields/treasury_yields.csv",
        help="Output CSV path (TSLib custom format)",
    )
    args = parser.parse_args()

    print("Downloading FRED DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS30 ...")
    df = build_panel()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n = len(df)
    n_test = int(n * 0.2)
    print(f"Wrote {out_path}  rows={n}  cols={list(df.columns)}")
    print(f"Range {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"70/10/20 test length ≈ {n_test}  (need > 720 for pred_len=720: {'OK' if n_test > 720 else 'TOO SHORT'})")


if __name__ == "__main__":
    main()
