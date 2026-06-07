"""
Collect JEPAVTS results into tables.

Parses the master results log (result_jepa_vts.txt) produced by
exp/exp_jepa_vts_long_term_forecasting.py and builds:

  1. results_detailed.csv : one row per (dataset, encoder, render, seq_len, pred_len)
                            with mse / mae / rmse.
  2. results_summary.csv  : one row per (dataset, encoder, render) with the
                            avg mse / avg mae across all horizons.

It also prints both tables to stdout.

Assumes model_id was built as: {dataset}_{render}_{seq_len}_{pred_len}_{single|dual}
(as in scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh). dataset may
itself contain underscores (e.g. exchange_rate, national_illness); parsing is done
from the right so that still works.

Usage:
    python utils/collect_results.py
    python utils/collect_results.py --results_file result_jepa_vts.txt --out_dir .
"""

import argparse
import os
import re
import sys
from collections import OrderedDict

# setting = long_term_forecast_{model_id}_{model}_{data}_ft..., model is fixed (JEPAVTS).
_MODEL_TOKEN = "JEPAVTS"
_SETTING_RE = re.compile(rf"^long_term_forecast_(?P<model_id>.+)_{_MODEL_TOKEN}_")
_METRIC_RE = re.compile(r"mse:([-\d.eE+]+),\s*mae:([-\d.eE+]+),\s*rmse:([-\d.eE+]+)")


def parse_model_id(model_id):
    """Parse {dataset}_{render}_{seq_len}_{pred_len}_{tag} from the right.

    Returns dict or None if it doesn't match the expected shape.
    """
    parts = model_id.split("_")
    if len(parts) < 5:
        return None
    tag = parts[-1]
    pred_len = parts[-2]
    seq_len = parts[-3]
    render = parts[-4]
    dataset = "_".join(parts[:-4])
    if tag not in ("single", "dual"):
        return None
    if not (seq_len.isdigit() and pred_len.isdigit()):
        return None
    return {
        "dataset": dataset,
        "encoder": tag,
        "render": render,
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
    }


def parse_results_file(path):
    """Parse result_jepa_vts.txt into a list of record dicts.

    The file is appended to over time, so a (setting) may appear multiple times;
    we keep the LAST occurrence (most recent run).
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    records = OrderedDict()  # keyed by full setting -> record (last wins)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = _SETTING_RE.match(line)
        if not m:
            i += 1
            continue
        setting = line
        meta = parse_model_id(m.group("model_id"))
        # find the next metric line within the next few lines
        metric = None
        for j in range(i + 1, min(i + 4, len(lines))):
            mm = _METRIC_RE.search(lines[j])
            if mm:
                metric = mm
                i = j
                break
        if metric and meta is not None:
            rec = dict(meta)
            rec["mse"] = float(metric.group(1))
            rec["mae"] = float(metric.group(2))
            rec["rmse"] = float(metric.group(3))
            records[setting] = rec
        i += 1

    return list(records.values())


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(_fmt(c) for c in r) + "\n")


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [len(h) for h in header]
    str_rows = [[_fmt(c) for c in r] for r in rows]
    for r in str_rows:
        for k, c in enumerate(r):
            widths[k] = max(widths[k], len(c))
    line = "  ".join(h.ljust(widths[k]) for k, h in enumerate(header))
    print(line)
    print("-" * len(line))
    for r in str_rows:
        print("  ".join(c.ljust(widths[k]) for k, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser(description="Collect JEPAVTS results into tables")
    ap.add_argument("--results_file", default="result_jepa_vts.txt",
                    help="Path to the master results log")
    ap.add_argument("--out_dir", default=".", help="Where to write CSV outputs")
    args = ap.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}", file=sys.stderr)
        print("   Run some experiments first (it is created/appended by the test loop).",
              file=sys.stderr)
        sys.exit(1)

    records = parse_results_file(args.results_file)
    if not records:
        print(f"⚠️  No parseable JEPAVTS records found in {args.results_file}.",
              file=sys.stderr)
        print("   (Expected model_id like ETTh1_RP_96_96_single.)", file=sys.stderr)
        sys.exit(1)

    # Sort for stable output
    records.sort(key=lambda r: (r["dataset"], r["render"], r["encoder"], r["pred_len"]))

    # --- Detailed table ---
    det_header = ["dataset", "encoder", "render", "seq_len", "pred_len", "mse", "mae", "rmse"]
    det_rows = [[r["dataset"], r["encoder"], r["render"], r["seq_len"],
                 r["pred_len"], r["mse"], r["mae"], r["rmse"]] for r in records]

    # --- Summary table: average over horizons per (dataset, encoder, render) ---
    groups = OrderedDict()
    for r in records:
        key = (r["dataset"], r["encoder"], r["render"])
        groups.setdefault(key, []).append(r)

    sum_header = ["dataset", "encoder", "render", "n_horizons", "avg_mse", "avg_mae"]
    sum_rows = []
    for (dataset, encoder, render), recs in groups.items():
        n = len(recs)
        avg_mse = sum(x["mse"] for x in recs) / n
        avg_mae = sum(x["mae"] for x in recs) / n
        sum_rows.append([dataset, encoder, render, n, avg_mse, avg_mae])
    sum_rows.sort(key=lambda x: (x[0], x[2], x[1]))

    det_path = os.path.join(args.out_dir, "results_detailed.csv")
    sum_path = os.path.join(args.out_dir, "results_summary.csv")
    write_csv(det_path, det_header, det_rows)
    write_csv(sum_path, sum_header, sum_rows)

    print_table("DETAILED (per horizon)", det_header, det_rows)
    print_table("SUMMARY (avg over horizons)", sum_header, sum_rows)
    print(f"\n✅ Wrote {det_path}  ({len(det_rows)} rows)")
    print(f"✅ Wrote {sum_path}  ({len(sum_rows)} rows)")


if __name__ == "__main__":
    main()
