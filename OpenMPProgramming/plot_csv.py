#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def main():
    ap = argparse.ArgumentParser(description="Plot GFLOP/s vs Threads from CSV")
    ap.add_argument("csvfile", help="Input CSV file with columns threads,gflops")
    ap.add_argument("--out", help="Output PNG filename (default: csv basename + .png)")
    args = ap.parse_args()

    df = pd.read_csv(args.csvfile)
    if not {"threads", "gflops"}.issubset(df.columns):
        raise ValueError("CSV must contain 'threads' and 'gflops' columns")

    df = df.sort_values("threads")

    plt.figure()
    plt.plot(df["threads"], df["gflops"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("GFLOP/s")
    plt.title(os.path.basename(args.csvfile))
    plt.grid(True)
    plt.tight_layout()

    outname = args.out if args.out else os.path.splitext(args.csvfile)[0] + ".png"
    plt.savefig(outname, dpi=200)
    print(f"Saved plot to {outname}")

if __name__ == "__main__":
    main()
