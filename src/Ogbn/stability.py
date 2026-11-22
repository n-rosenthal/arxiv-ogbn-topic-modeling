"""Topic stability across seeds (and across models), as a number instead of an anecdote.

    python -m src.Ogbn.stability --root data/ogbn --emb minilm --topics 50 --models prodlda ctm ctm_knn ctm_both

For each model, the topics of every pair of seeds are matched one-to-one (Hungarian, Jaccard of the top-10 words).
Reported per model: mean matched Jaccard over all seed pairs (+- sd over pairs) and the share of matched topics with
Jaccard >= 0.3. For each model != --ref also the same statistics against --ref (same seed): if "model vs ref" is as
low as "ref vs ref across seeds", the graph is not changing the topics beyond what a re-run already changes.
Writes <results>/stability_K<K>.org.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from . import aggregate
from .compare_topics import match_topics

THRESH = 0.3


def _words(d: Path, model: str, K: int, seed: int, tag: str = ""):
    f = d / f"{model}_K{K}{('_' + tag) if tag else ''}_s{seed}.json"
    return json.loads(f.read_text())["top_words"] if f.exists() else None


def _stats(pairs):
    j = np.array([np.mean([m[2] for m in match]) for match in pairs])
    c = np.array([np.mean([m[2] >= THRESH for m in match]) for match in pairs])
    return j, c


def stability(d: Path, K: int, models: list[str], ref: str, seeds: list[int]) -> list[dict]:
    rows = []
    W = {m: {s: _words(d, m, K, s) for s in seeds} for m in set(models) | {ref}}
    for m in models:
        ss = [s for s in seeds if W[m].get(s) is not None]
        within = [match_topics(W[m][a], W[m][b])[0] for a, b in combinations(ss, 2)]
        row = {"modelo": aggregate.NAMES.get(m, m), "pares de sementes": len(within)}
        if within:
            j, c = _stats(within)
            row.update({"Jaccard médio (entre sementes)": f"{j.mean():.3f} ± {j.std(ddof=1) if len(j) > 1 else 0:.3f}",
                        f"tópicos com J≥{THRESH}": f"{100 * c.mean():.0f}%"})
        if m != ref:
            cross = [match_topics(W[m][s], W[ref][s])[0] for s in ss if W[ref].get(s) is not None]
            if cross:
                j, c = _stats(cross)
                row.update({f"Jaccard vs {aggregate.NAMES.get(ref, ref)} (mesma semente)":
                            f"{j.mean():.3f} ± {j.std(ddof=1) if len(j) > 1 else 0:.3f}",
                            f"J≥{THRESH} vs ref": f"{100 * c.mean():.0f}%"})
        rows.append(row)
    return rows


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["lda", "nmf", "prodlda", "ctm", "ctm_knn", "ctm_both"])
    ap.add_argument("--ref", default="ctm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    d = aggregate.results_dir(a.root, a.emb)
    t = pd.DataFrame(stability(d, a.topics, a.models, a.ref, a.seeds)).fillna("")
    (d / f"stability_K{a.topics}.org").write_text(aggregate.to_org(t))
    print(t.to_string(index=False))
