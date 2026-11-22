"""Step 7 -- collect results into tables for the paper.

    python -m src.Ogbn.aggregate --root data/ogbn --emb minilm --topics 50 --ref ctm
Writes <root>/results/<emb>/table_K<K>.{csv,tex,org} and prints a paired comparison against --ref.
mean +- std over seeds. The paired difference uses the same seed index in both models and a t-based 95% CI;
with 5 seeds this is indicative, not a significance proof (say so in the paper).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import data

NAMES = {"lda": "LDA", "nmf": "NMF", "bertopic": "BERTopic", "prodlda": "ProdLDA", "ctm": "CTM",
         "ctm_knn": "CTM + kNN", "ctm_cite": "CTM + citação", "ctm_both": "CTM + kNN + citação",
         "ctm_rand": "CTM + vizinhos aleatórios", "zeroshot": "Só embedding", "zeroshot_both": "Só embedding + grafo"}
ORDER = list(NAMES)
COLS = [("npmi", "NPMI"), ("diversity", "Diversidade"), ("nmi_test", "NMI"), ("acc_test", "Acc."),
        ("f1_test", "F1-macro"), ("n_active", "Tópicos ativos")]


def results_dir(root: str, emb: str) -> Path:
    return Path(root) / "results" / emb / data.load(root, with_bow=False).fingerprint


def load(root: str, emb: str, K: int, tag: str = "") -> pd.DataFrame:
    rows = []
    for f in results_dir(root, emb).glob(f"*_K{K}*_s*.json"):
        r = json.loads(f.read_text())
        if r["K"] != K or r.get("tag", "") != tag:
            continue
        rows.append({"model": r["model"], "seed": r["seed"], "elapsed_min": r["elapsed_s"] / 60, **r["metrics"]})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in [m for m in ORDER if m in set(df["model"])]:
        d = df[df["model"] == m]
        row = {"Modelo": NAMES[m], "seeds": len(d)}
        for key, name in COLS + [("elapsed_min", "Tempo (min)")]:
            row[name] = f"{d[key].mean():.3f} ± {d[key].std(ddof=1) if len(d) > 1 else 0:.3f}" \
                if key not in ("n_active", "elapsed_min") else f"{d[key].mean():.1f}"
        out.append(row)
    return pd.DataFrame(out)


def paired(df: pd.DataFrame, ref: str, key: str = "npmi") -> pd.DataFrame:
    r = df[df["model"] == ref].set_index("seed")[key]
    rows = []
    for m in [m for m in ORDER if m in set(df["model"]) and m != ref]:
        c = df[df["model"] == m].set_index("seed")[key]
        d = (c - r).dropna()
        if len(d) < 2:
            continue
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"modelo": NAMES[m], f"Δ{key} vs {NAMES[ref]}": f"{d.mean():+.4f}",
                     "IC95%": f"[{d.mean() - half:+.4f}, {d.mean() + half:+.4f}]", "n": len(d)})
    return pd.DataFrame(rows)


# (hypothesis, comparison label, model, reference, metric, kind)  kind: "diff" or "gap"
HYP = [("H1", "CTM - ProdLDA", "ctm", "prodlda", "nmi_test", "diff"),
       ("H1", "CTM - ProdLDA", "ctm", "prodlda", "acc_test", "diff"),
       ("H1", "CTM - ProdLDA", "ctm", "prodlda", "npmi", "diff"),
       ("H2", "CTM+kNN - CTM", "ctm_knn", "ctm", "nmi_test", "diff"),
       ("H2", "CTM+kNN - CTM", "ctm_knn", "ctm", "acc_test", "diff"),
       ("H2", "CTM+citação - CTM", "ctm_cite", "ctm", "nmi_test", "diff"),
       ("H2", "CTM+citação - CTM", "ctm_cite", "ctm", "acc_test", "diff"),
       ("H2", "CTM+ambos - CTM", "ctm_both", "ctm", "nmi_test", "diff"),
       ("H2", "CTM+ambos - CTM", "ctm_both", "ctm", "acc_test", "diff"),
       ("H2", "CTM+kNN - CTM (NPMI)", "ctm_knn", "ctm", "npmi", "diff"),
       ("H3", "citação - kNN", "ctm_cite", "ctm_knn", "nmi_test", "diff"),
       ("H3", "citação - kNN", "ctm_cite", "ctm_knn", "acc_test", "diff"),
       ("H3", "ambos - kNN", "ctm_both", "ctm_knn", "nmi_test", "diff"),
       ("H3", "ambos - kNN", "ctm_both", "ctm_knn", "acc_test", "diff"),
       ("H3", "ambos - citação", "ctm_both", "ctm_cite", "nmi_test", "diff"),
       ("H3", "ambos - citação", "ctm_both", "ctm_cite", "acc_test", "diff"),
       ("controle", "kNN - vizinhos aleatórios", "ctm_knn", "ctm_rand", "nmi_test", "diff"),
       ("controle", "kNN - vizinhos aleatórios", "ctm_knn", "ctm_rand", "acc_test", "diff"),
       ("controle", "aleatórios - CTM", "ctm_rand", "ctm", "nmi_test", "diff"),
       ("controle", "aleatórios - CTM", "ctm_rand", "ctm", "acc_test", "diff"),
       ("extra", "só embedding+grafo - só embedding", "zeroshot_both", "zeroshot", "nmi_test", "diff"),
       ("extra", "só embedding+grafo - só embedding", "zeroshot_both", "zeroshot", "acc_test", "diff"),
       ("extra", "só embedding+grafo - CTM+ambos", "zeroshot_both", "ctm_both", "nmi_test", "diff"),
       ("extra", "só embedding+grafo - CTM+ambos", "zeroshot_both", "ctm_both", "acc_test", "diff"),
       ("extra", "só embedding - CTM", "zeroshot", "ctm", "nmi_test", "diff"),
       ("extra", "só embedding - CTM", "zeroshot", "ctm", "acc_test", "diff"),
       ("H4", "queda val→teste (NMI): kNN - CTM", "ctm_knn", "ctm", "nmi", "gap"),
       ("H4", "queda val→teste (NMI): citação - CTM", "ctm_cite", "ctm", "nmi", "gap"),
       ("H4", "queda val→teste (NMI): ambos - CTM", "ctm_both", "ctm", "nmi", "gap"),
       ("H4", "queda val→teste (Acc): kNN - CTM", "ctm_knn", "ctm", "acc", "gap"),
       ("H4", "queda val→teste (Acc): citação - CTM", "ctm_cite", "ctm", "acc", "gap"),
       ("H4", "queda val→teste (Acc): ambos - CTM", "ctm_both", "ctm", "acc", "gap")]


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down), returned in the input order."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    """Paired (by seed) differences for the pre-specified comparisons. gap = metric_val - metric_test.
    p = two-sided paired t-test; p_Holm = Holm correction applied within each family (column H)."""
    def series(model, metric, kind):
        d = df[df["model"] == model].set_index("seed")
        return d[f"{metric}_val"] - d[f"{metric}_test"] if kind == "gap" else d[metric]
    rows = []
    for h, label, m, ref, metric, kind in HYP:
        if m not in set(df["model"]) or ref not in set(df["model"]):
            continue
        d = (series(m, metric, kind) - series(ref, metric, kind)).dropna()
        if len(d) < 2:
            continue
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - half, d.mean() + half
        pval = float(stats.ttest_1samp(d, 0.0).pvalue) if d.std(ddof=1) > 0 else 1.0
        rows.append({"H": h, "comparação": label, "métrica": metric if kind == "diff" else f"gap {metric}",
                     "Δ média": f"{d.mean():+.4f}", "IC95%": f"[{lo:+.4f}, {hi:+.4f}]",
                     "IC exclui 0": "sim" if lo > 0 or hi < 0 else "não", "n": len(d), "_p": pval})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p"] = out["_p"].map(lambda v: f"{v:.4f}")
    adj = pd.Series(index=out.index, dtype=float)
    for _, idx in out.groupby("H").groups.items():
        adj.loc[idx] = holm(out.loc[idx, "_p"].tolist())
    out["p Holm"] = adj.map(lambda v: f"{v:.4f}")
    out["sig. após Holm"] = np.where(adj < 0.05, "sim", "não")
    return out.drop(columns="_p")


def to_org(t: pd.DataFrame) -> str:
    head = "| " + " | ".join(t.columns) + " |\n|" + "+".join("-" * (len(c) + 2) for c in t.columns) + "|\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in t.itertuples(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ref", default="ctm")
    a = ap.parse_args()
    df = load(a.root, a.emb, a.topics, a.tag)
    if df.empty:
        raise SystemExit("no results found")
    t = table(df)
    base = results_dir(a.root, a.emb) / f"table_K{a.topics}{('_' + a.tag) if a.tag else ''}"
    t.to_csv(str(base) + ".csv", index=False)
    Path(str(base) + ".tex").write_text(t.to_latex(index=False, escape=True).replace("±", "$\\pm$"))
    Path(str(base) + ".org").write_text(to_org(t))
    print(t.to_string(index=False))
    hyp = hypotheses(df)
    if not hyp.empty:
        Path(str(base).replace("table_", "hypotheses_") + ".org").write_text(to_org(hyp))
        print("\nHipóteses (diferenças pareadas por semente):\n" + hyp.to_string(index=False))
    for key in ("npmi", "nmi_test", "acc_test"):
        p = paired(df, a.ref, key)
        if not p.empty:
            print("\n" + p.to_string(index=False))
