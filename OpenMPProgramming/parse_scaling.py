#!/usr/bin/env python3
import argparse, re, os
from collections import OrderedDict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HDR  = re.compile(r"^===\s*(?P<mode>.+?)\s*\|\s*OMP_THREADS=(?P<t>\d+)\s*===")
RATE = re.compile(r"Rate:\s*(?P<gflops>[0-9]+(?:\.[0-9]+)?)\s*GFLOP/s")

def parse(path, label):
    rows=[]; mode=None; t=None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m=HDR.search(line)
            if m: mode=m.group("mode").strip(); t=int(m.group("t")); continue
            r=RATE.search(line)
            if r and mode is not None and t is not None:
                rows.append({"mode":mode,"threads":t,"gflops":float(r.group("gflops")),"label":label})
                mode=None; t=None
    return rows

def plot(df_mode, outpng):
    labels=list(OrderedDict.fromkeys(df_mode["label"].tolist()))
    plt.figure()
    for lab in labels:
        d=df_mode[df_mode["label"]==lab].sort_values("threads")
        plt.plot(d["threads"], d["gflops"], marker="o", label=lab)
    plt.xlabel("Threads"); plt.ylabel("GFLOP/s"); plt.title(df_mode["mode"].iloc[0])
    plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--input", action="append", default=[], help="label=path")
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows=[]
    for spec in args.input:
        assert "=" in spec, "--input must be label=path"
        lab, path = spec.split("=",1)
        rows += parse(path, lab)
    if not rows: raise SystemExit("No data parsed.")
    df=pd.DataFrame(rows)
    for mode, d in df.groupby("mode"):
        safe=mode.lower().replace(" ","_").replace("(","").replace(")","").replace("/","_")
        d.sort_values(["label","threads"]).to_csv(os.path.join(args.outdir,f"{safe}_combined.csv"), index=False)
        for lab,dl in d.groupby("label"):
            dl.sort_values("threads").to_csv(os.path.join(args.outdir,f"{safe}_{lab}.csv"), index=False)
        plot(d, os.path.join(args.outdir,f"{safe}_scaling.png"))
    print("Wrote:", args.outdir)
if __name__=="__main__": main()
