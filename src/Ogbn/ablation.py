"""Sensitivity of the main comparison to K (number of topics) and k (kNN neighbours).

    python -m src.Ogbn.ablation --root data/ogbn --emb minilm --Ks 20 50 100 --ks 5 10 20
Reads results written by run_all.sh (K sweep: K=20/100, no tag) and by run_experiment --k <k> --tag k<k> (k sweep).
For each K: mean +- sd of NMI/accuracy/NPMI for CTM and CTM+kNN and the paired difference (kNN - CTM) with a t-based 95% CI.
For each k: CTM+kNN (K=50) against CTM (K=50). k=10 uses the main runs (no tag).
Writes <results>/ablation_K.org and ablation_k.org.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import aggregate

METRICS = [("nmi_test", "NMI"), ("acc_test", "Acc."), ("npmi", "NPMI")]


def _ms(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def _diff(a: pd.Series, b: pd.Series) -> str:
    d = (a - b).dropna()
    if len(d) < 2:
        return "—"
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} [{d.mean() - h:+.4f}; {d.mean() + h:+.4f}]"


def table_K(root: str, emb: str, Ks: list[int], model: str = "ctm_knn", ref: str = "ctm") -> pd.DataFrame:
    rows = []
    for K in Ks:
        df = aggregate.load(root, emb, K)
        if df.empty or ref not in set(df["model"]) or model not in set(df["model"]):
            continue
        r, m = df[df["model"] == ref].set_index("seed"), df[df["model"] == model].set_index("seed")
        row = {"K": K, "sementes": len(r.index.intersection(m.index))}
        for key, name in METRICS:
            row[f"{name} {aggregate.NAMES[ref]}"] = _ms(r[key])
            row[f"{name} {aggregate.NAMES[model]}"] = _ms(m[key])
            row[f"Δ{name} [IC95%]"] = _diff(m[key], r[key])
        rows.append(row)
    return pd.DataFrame(rows)


def table_k(root: str, emb: str, K: int, ks: list[int], model: str = "ctm_knn", ref: str = "ctm", main_k: int = 10) -> pd.DataFrame:
    r = aggregate.load(root, emb, K)
    r = r[r["model"] == ref].set_index("seed")
    rows = []
    for k in ks:
        df = aggregate.load(root, emb, K, "" if k == main_k else f"k{k}")
        m = df[df["model"] == model].set_index("seed")
        if m.empty or r.empty:
            continue
        row = {"k": k, "sementes": len(r.index.intersection(m.index))}
        for key, name in METRICS:
            row[f"{name}"] = _ms(m[key])
            row[f"Δ{name} vs CTM [IC95%]"] = _diff(m[key], r[key])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--Ks", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--K", type=int, default=50)
    a = ap.parse_args()
    d = aggregate.results_dir(a.root, a.emb)
    for name, t in (("K", table_K(a.root, a.emb, a.Ks)), ("k", table_k(a.root, a.emb, a.K, a.ks))):
        if t.empty:
            print(f"[{name}] no results yet")
            continue
        (d / f"ablation_{name}.org").write_text(aggregate.to_org(t))
        print(f"\n== sensibilidade a {name}\n" + t.to_string(index=False))
