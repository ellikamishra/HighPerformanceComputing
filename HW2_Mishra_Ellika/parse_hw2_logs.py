#!/usr/bin/env python3
"""
Parse HW2 CUDA logs to a single CSV + summary tables (and optional plots).

salloc -C cpu -q debug -t 03:00:00 -N 1
salloc -C cpu -q debug -t 00:30:00 -N 1
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PREFIX = "CSV,"


def _maybe_int(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_csv_line(line):
    if not line.startswith(CSV_PREFIX):
        return {}
    parts = line[len(CSV_PREFIX):].split(",")
    record = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in {"M", "N", "K", "L", "D", "TILE"}:
            record[k] = _maybe_int(v)
        elif k in {"ms", "GF", "diff"}:
            record[k] = _maybe_float(v)
        elif k == "BLOCK":
            record[k] = v
            if "x" in v:
                bx, by = v.split("x", 1)
                record["BX"], record["BY"] = _maybe_int(bx), _maybe_int(by)
        else:
            record[k] = v
    return record

def extractrecords(Files):
    recs = []
    for f in Files:
        text = Path(f).read_text(errors="ignore")
        for line in text.splitlines():
            if line.startswith(CSV_PREFIX):
                rec = parse_csv_line(line)
                if rec:
                    rec["source"] = str(f)
                    recs.append(rec)
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Log Files/dirs to scan")
    ap.add_argument("--plot", action="store_true", help="Also create PNG plots")
    ap.add_argument("-o","--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    # collect Files
    Files = []
    for p in args.paths:
        P = Path(p)
        if P.is_dir():
            Files.extend([f for f in P.rglob("*") if f.is_file()])
        elif P.is_file():
            Files.append(P)
    Files = sorted(set(Files))

    recs = extractrecords(Files)
    if not recs:
        print("No CSV lines found")
        return

    df = pd.DataFrame(recs)
    numeric_cols = [c for c in ["M", "N", "K", "L", "D", "TILE", "BX", "BY", "ms", "GF", "diff"] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    outdir = Path(args.outdir)
    if outdir.exists() and not outdir.is_dir():
        raise RuntimeError(f"Output path {outdir} exists and is not a directory")
    outdir.mkdir(exist_ok=True, parents=True)

    # merged results
    df.to_csv(outdir/"merged_results.csv", index=False)
    print(f"[OK] merged_results.csv with {len(df)} rows")

    # summary
    summary = (df.groupby(["ALG","TILE","BLOCK"], dropna=False)
                 .agg(GF_mean=("GF","mean"),
                      GF_std=("GF","std"),
                      ms_mean=("ms","mean"),
                      runs=("ALG","count"))
                 .reset_index())
    summary.to_csv(outdir/"summary_by_alg.csv", index=False)
    print("[OK] summary_by_alg.csv written")

    if args.plot:
        plots = outdir/"plots"
        plots.mkdir(exist_ok=True)
        # Example: performance vs TILE
        for alg, sub in df.groupby("ALG"):
            if "TILE" in sub and sub["TILE"].notna().any():
                plt.figure()
                plt.plot(sub["TILE"], sub["GF"], "o-")
                plt.xlabel("TILE")
                plt.ylabel("GFLOP/s")
                plt.title(f"{alg} performance vs TILE")
                plt.grid(True)
                plt.savefig(plots/f"{alg}_vs_TILE.png", dpi=150)
                plt.close()

if __name__ == "__main__":
    main()
