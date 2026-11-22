"""Qualitative comparison of the topics of two runs (two models, or the same model with two seeds).

    python -m src.Ogbn.compare_topics --emb minilm --topics 50 --a ctm --b ctm_both --seed 0
    python -m src.Ogbn.compare_topics --emb minilm --topics 50 --a ctm --b ctm --seed 0 --seed-b 1   # stability across seeds

Topics of A and B are matched one-to-one (Hungarian algorithm) by Jaccard overlap of their top-10 words.
Output (printed and saved as org next to the results): summary, matched pairs sorted by the NPMI change
(B - A), and topics of each model with no close counterpart.
If the runs were made with --save-theta, each topic also gets its dominant arXiv category
(among test documents whose argmax topic it is) and the median publication year.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import aggregate, data, evaluate


def _load(root: str, emb: str, model: str, K: int, seed: int, tag: str = "") -> tuple[dict, Path]:
    d = aggregate.results_dir(root, emb)
    path = d / f"{model}_K{K}{('_' + tag) if tag else ''}_s{seed}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return json.loads(path.read_text()), path


def jaccard_matrix(A: list[list[str]], B: list[list[str]]) -> np.ndarray:
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        sa = set(a)
        for j, b in enumerate(B):
            sb = set(b)
            M[i, j] = len(sa & sb) / len(sa | sb)
    return M


def match_topics(A: list[list[str]], B: list[list[str]]):
    J = jaccard_matrix(A, B)
    r, c = linear_sum_assignment(-J)
    return [(int(i), int(j), float(J[i, j])) for i, j in zip(r, c)], J


def topic_profile(theta_path: Path, nodes) -> dict:
    """topic -> (dominant category, share of the topic's docs, n docs, median year) on the test split."""
    th = np.load(theta_path)["test"].astype(np.float32)
    test = nodes[nodes["split"] == 2].reset_index(drop=True)
    arg = th.argmax(1)
    out = {}
    for k in range(th.shape[1]):
        sel = test[arg == k]
        if len(sel) == 0:
            out[k] = ("-", 0.0, 0, float("nan"))
            continue
        cat, n = Counter(sel["label_name"]).most_common(1)[0]
        out[k] = (cat, n / len(sel), len(sel), float(sel["year"].median()))
    return out


def run(root: str, emb: str, K: int, a: str, b: str, seed: int, seed_b: int | None, min_jaccard: float,
        tag: str = "") -> str:
    seed_b = seed if seed_b is None else seed_b
    ra, pa = _load(root, emb, a, K, seed, tag)
    rb, pb = _load(root, emb, b, K, seed_b, tag)
    bundle = data.load(root)
    w2i = {w: i for i, w in enumerate(bundle.vocab)}
    Xbin = evaluate.binary_csc(bundle.bow)

    def npmi(tw):
        return evaluate.npmi_per_topic(np.array([[w2i[w] for w in t] for t in tw]), Xbin)

    TA, TB = ra["top_words"], rb["top_words"]
    nA, nB = npmi(TA), npmi(TB)
    pairs, J = match_topics(TA, TB)
    profA = profB = None
    ta, tb = pa.with_name(pa.stem + "_theta.npz"), pb.with_name(pb.stem + "_theta.npz")
    if ta.exists() and tb.exists():
        profA, profB = topic_profile(ta, bundle.nodes), topic_profile(tb, bundle.nodes)

    def desc(prof, k):
        if prof is None:
            return ""
        cat, share, n, yr = prof[k]
        return f" [{cat} {share:.0%}, n={n}, ano mediano {yr:.0f}]"

    good = [p for p in pairs if p[2] >= min_jaccard]
    lines = [f"* {a} (semente {seed}) vs {b} (semente {seed_b}), K={K}", "",
             f"- tópicos com par próximo (Jaccard >= {min_jaccard}): {len(good)}/{K}",
             f"- Jaccard médio dos pares: {np.mean([p[2] for p in pairs]):.3f}",
             f"- NPMI médio: {a if a != b else f'{a} (s{seed})'} {nA.mean():.3f} | {b if a != b else f'{b} (s{seed_b})'} {nB.mean():.3f}", ""]
    lines += ["** Pares alinhados (ordenados por ΔNPMI = B - A)", "",
              f"| Jaccard | NPMI A | NPMI B | Δ | palavras A | palavras B |", "|---+---+---+---+---+---|"]
    for i, j, jac in sorted(pairs, key=lambda p: nB[p[1]] - nA[p[0]]):
        lines.append(f"| {jac:.2f} | {nA[i]:.2f} | {nB[j]:.2f} | {nB[j] - nA[i]:+.2f} | "
                     f"{' '.join(TA[i])}{desc(profA, i)} | {' '.join(TB[j])}{desc(profB, j)} |")
    la, lb = (a, b) if a != b else (f"{a} (semente {seed})", f"{b} (semente {seed_b})")   # a == b: same model, 2 seeds
    best = {la: J.max(axis=1), lb: J.max(axis=0)}        # best Jaccard of each topic against the other run
    for name, T, n_, prof in ((la, TA, nA, profA), (lb, TB, nB, profB)):
        ids = [k for k in range(len(T)) if best[name][k] < 0.2]
        lines += ["", f"** Tópicos de {name} sem contraparte próxima (Jaccard máx < 0.2): {len(ids)}", ""]
        lines += [f"- ({n_[k]:.2f}) {' '.join(T[k])}{desc(prof, k)}" for k in ids]
    text = "\n".join(lines) + "\n"
    out = pa.parent / f"qualitativa_{a}_vs_{b}_K{K}_s{seed}{'' if seed_b == seed else f'_vs_s{seed_b}'}.org"
    out.write_text(text)
    print(text)
    print(f"[saved] {out}")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-b", type=int, default=None)
    ap.add_argument("--min-jaccard", type=float, default=0.3)
    ap.add_argument("--tag", default="")
    x = ap.parse_args()
    run(x.root, x.emb, x.topics, x.a, x.b, x.seed, x.seed_b, x.min_jaccard, x.tag)
