import sys, json
sys.path.insert(0, "drift")
import numpy as np, pandas as pd
from drift_math import make_bins, bin_proportions, psi

ref   = pd.read_parquet("model/reference_sample.parquet")
df    = pd.read_csv("data/creditcard.csv")
feats = [c for c in json.load(open("model/metadata.json"))["features"] if c != "Time"]

edges = {f: make_bins(ref[f]) for f in feats}
props = {f: bin_proportions(ref[f], edges[f]) for f in feats}

def report(name, cur):
    p = {f: psi(props[f], bin_proportions(cur[f].to_numpy(), edges[f])) for f in feats}
    top = max(p, key=p.get)
    print(f"{name:28s} max PSI={p[top]:.4f} ({top})  drifted={sum(v > 0.25 for v in p.values())}")

report("random 3000 rows",        df.sample(3000, random_state=1))
report("first 3000 rows (seq.)",  df.iloc[:3000])
report("3000 rows @ offset 100k", df.iloc[100_000:103_000])
report("3000 rows @ offset 200k", df.iloc[200_000:203_000])