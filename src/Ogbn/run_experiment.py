"""Step 6 -- train/evaluate one model for several seeds; one JSON per (model, K, seed).

    python -m src.Ogbn.run_experiment --model lda      --topics 50 --seeds 0 1 2 3 4
    python -m src.Ogbn.run_experiment --model ctm_knn  --topics 50 --seeds 0 1 2 3 4 --emb minilm --k 10

models: lda | nmf | bertopic | prodlda | ctm | ctm_knn | ctm_cite | ctm_both | ctm_rand | zeroshot | zeroshot_both
Results: <root>/results/<emb>/<corpus-fingerprint>/<model>_K<K>[_tag]_s<seed>.json   (re-running skips existing files; --force overrides)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import baselines, data, evaluate, graphs

NEURAL = {"prodlda": ("none", False, True), "ctm": ("none", True, True), "ctm_knn": ("knn", True, True),
          "ctm_cite": ("cite", True, True), "ctm_both": ("both", True, True),
          # controls / extra experiments
          "ctm_rand": ("rand", True, True),            # random neighbours: is the gain really from neighbour information?
          "zeroshot": ("none", True, False),           # embedding only, no BoW at the encoder (ZeroShotTM style)
          "zeroshot_both": ("both", True, False)}      # embedding + kNN/citation graph, no BoW at the encoder
ALL = ["lda", "nmf", "bertopic", *NEURAL]


def labels_by_split(b: data.Bundle) -> dict:
    return {n: b.labels[b.split == s] for s, n in enumerate(data.SPLITS)}


def run(root: str, model: str, K: int, seeds: list[int], emb: str, k: int, hops: int, tag: str,
        force: bool, cfg_overrides: dict, save_theta: bool = False) -> None:
    b = data.load(root)
    if b.bow is None:
        raise SystemExit("Run text_features first (bow.npz missing).")
    out_dir = Path(root) / "results" / emb / b.fingerprint
    out_dir.mkdir(parents=True, exist_ok=True)
    Xbin = evaluate.binary_csc(b.bow)
    labels = labels_by_split(b)
    need_emb = model == "bertopic" or (model in NEURAL and NEURAL[model][1])
    E = data.load_embeddings(root, emb) if need_emb else None
    if E is not None and E.shape[0] != b.n:
        raise SystemExit(f"emb_{emb}.npy has {E.shape[0]} rows but the corpus has {b.n}: rerun the embeddings step.")

    feats, gstats = None, {}
    if model in NEURAL and NEURAL[model][1]:
        kind = NEURAL[model][0]
        A = graphs.build_adjacency(kind, E, b.edges, b.split, k, Path(root) / "graphs", f"{emb}_{b.fingerprint}")
        gstats = graphs.graph_stats(A, b.split)
        feats = graphs.propagate(E, A, hops)
        print(f"[graph:{kind}] {gstats}  feature dim {feats.shape[1]}")

    for seed in seeds:
        path = out_dir / f"{model}_K{K}{('_' + tag) if tag else ''}_s{seed}.json"
        if path.exists() and not force:
            print(f"[skip] {path.name}")
            continue
        print(f"[run ] {model} K={K} seed={seed}")
        t0 = time.time()
        info: dict = {}
        if model == "lda":
            top_idx, theta = baselines.run_lda(b.bow, b.split, K, seed)
        elif model == "nmf":
            top_idx, theta = baselines.run_nmf(b.bow, b.split, K, seed)
        elif model == "bertopic":
            top_idx, theta = baselines.run_bertopic(b.texts, E, b.split, b.vocab, K, seed)
        else:
            from . import ctm                                   # torch is only needed here
            cfg = ctm.Config(topics=K, **cfg_overrides)
            _, use_emb, use_bow = NEURAL[model]
            top_idx, theta, info = ctm.fit_predict(b.bow, b.split, feats, use_bow, cfg, seed)
        elapsed = time.time() - t0
        metrics = evaluate.evaluate(top_idx, theta, labels, Xbin, seed=seed)
        res = {"model": model, "K": K, "seed": seed, "emb": emb, "tag": tag, "k_nn": k, "hops": hops,
               "fingerprint": b.fingerprint, "n_nodes": b.n, "metrics": metrics, "elapsed_s": elapsed, "info": info, "graph": gstats,
               "config": cfg_overrides,
               "top_words": [[b.vocab[i] for i in row] for row in top_idx]}
        path.write_text(json.dumps(res, indent=1))
        if save_theta:                                  # needed by compare_topics for category / year analysis
            np.savez_compressed(path.with_name(path.stem + "_theta.npz"), test=theta["test"].astype(np.float16))
        print("       " + "  ".join(f"{k_}={v:.3f}" if isinstance(v, float) else f"{k_}={v}"
                                    for k_, v in metrics.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--model", required=True, choices=ALL)
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--k", type=int, default=10, help="kNN neighbours")
    ap.add_argument("--hops", type=int, default=1, help="propagation hops (1 or 2)")
    ap.add_argument("--tag", default="", help="suffix to keep ablation runs apart")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-theta", action="store_true", help="also store test-set theta (float16) per run")
    # neural hyper-parameters (all optional)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--theta-dropout", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--kl-weight", type=float)
    ap.add_argument("--kl-anneal", type=int)
    ap.add_argument("--no-decoder-bn", dest="decoder_bn", action="store_false", default=None)
    a = ap.parse_args()
    over = {k_: v for k_, v in dict(hidden=a.hidden, dropout=a.dropout, theta_dropout=a.theta_dropout, lr=a.lr,
                                    batch=a.batch, epochs=a.epochs, kl_weight=a.kl_weight,
                                    kl_anneal=a.kl_anneal, decoder_bn=a.decoder_bn).items() if v is not None}
    run(a.root, a.model, a.topics, a.seeds, a.emb, a.k, a.hops, a.tag, a.force, over, a.save_theta)
