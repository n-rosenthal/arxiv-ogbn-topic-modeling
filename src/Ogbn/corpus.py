"""Step 2 -- join ogbn-arxiv graph + raw text into one clean corpus.

    python -m src.Ogbn.corpus --root data/ogbn [--max-nodes 60000] [--seed 0]

Writes data/ogbn/corpus/:
    nodes.csv.gz   idx, paper_id, year, label, label_name, split (0 train / 1 val / 2 test), title, abstract
    edges.npy      int32 (E, 2); row (i, j) means "paper i cites paper j"; indices refer to nodes.csv.gz
    meta.json      statistics

Official OGB time split: train <= 2017, validation 2018, test >= 2019.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _find(raw_dir: Path, pattern: str) -> Path | None:
    hits = sorted(raw_dir.rglob(pattern))
    return hits[0] if hits else None


def _read_numeric(path: Path, ncols: int | None = None) -> pd.DataFrame:
    """Read a header-less (or headed) numeric csv, dropping a header row if present."""
    df = pd.read_csv(path, header=None, dtype=str)
    first = df.iloc[0].astype(str)
    if not all(re.fullmatch(r"-?\d+", v.strip()) for v in first):
        df = df.iloc[1:]
    df = df.astype(np.int64)
    return df if ncols is None else df.iloc[:, :ncols]


def load_ogb(raw_dir: Path) -> dict:
    edge_p = _find(raw_dir, "edge.csv.gz")
    year_p = _find(raw_dir, "node_year.csv.gz")
    label_p = _find(raw_dir, "node-label.csv.gz")
    map_p = _find(raw_dir, "nodeidx2paperid.csv.gz")
    for name, p in [("edge.csv.gz", edge_p), ("node_year.csv.gz", year_p),
                    ("node-label.csv.gz", label_p), ("nodeidx2paperid.csv.gz", map_p)]:
        if p is None:
            raise SystemExit(f"{name} not found under {raw_dir}. Did python -m src.Ogbn.download finish?")
    edges = _read_numeric(edge_p, 2).to_numpy()
    year = _read_numeric(year_p, 1).to_numpy().ravel()
    label = _read_numeric(label_p, 1).to_numpy().ravel()
    mapping = _read_numeric(map_p, 2).to_numpy()
    n = len(year)
    node_idx, paper_id = mapping[:, 0], mapping[:, 1]
    order = np.argsort(node_idx)
    node_idx, paper_id = node_idx[order], paper_id[order]
    if len(node_idx) != n or not np.array_equal(node_idx, np.arange(n)):
        raise SystemExit("nodeidx2paperid does not cover 0..N-1; unexpected OGB layout.")

    split = {}
    for key, name in [(0, "train"), (1, "valid"), (2, "test")]:
        p = _find(raw_dir / "arxiv" if (raw_dir / "arxiv").exists() else raw_dir, f"split/time/{name}.csv.gz")
        if p is None:
            p = _find(raw_dir, f"time/{name}.csv.gz")
        split[key] = _read_numeric(p, 1).to_numpy().ravel() if p is not None else None
    split_arr = np.full(n, -1, dtype=np.int8)
    if all(v is not None for v in split.values()):
        for key, idx in split.items():
            split_arr[idx] = key
    else:
        print("[warn] official split files missing; using year rule (<=2017 / 2018 / >=2019)")
        split_arr = np.where(year <= 2017, 0, np.where(year == 2018, 1, 2)).astype(np.int8)
    assert (split_arr >= 0).all()

    names = None
    lp = _find(raw_dir, "labelidx2*.csv.gz")
    if lp is not None:
        df = pd.read_csv(lp, dtype=str)
        names = {int(a): str(b) for a, b in zip(df.iloc[:, 0], df.iloc[:, 1])}
    return dict(edges=edges, year=year, label=label, paper_id=paper_id, split=split_arr, label_names=names)


def load_titleabs(path: Path) -> pd.DataFrame:
    """titleabs.tsv(.gz): paper id <TAB> title <TAB> abstract (header optional)."""
    df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE, dtype=str,
                     on_bad_lines="skip", keep_default_na=False, encoding_errors="replace")
    if df.shape[1] < 3:
        raise SystemExit(f"{path}: expected 3 tab-separated columns, found {df.shape[1]}")
    df = df.iloc[:, :3]
    df.columns = ["paper_id", "title", "abstract"]
    df = df[df["paper_id"].str.fullmatch(r"\d+")]          # drops a header row, if any
    df["paper_id"] = df["paper_id"].astype(np.int64)
    return df.drop_duplicates("paper_id")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build(root: str | Path = "data/ogbn", min_abstract_words: int = 30,
          max_nodes: int | None = None, seed: int = 0, raw_dir: str | Path | None = None,
          text_mode: str = "full") -> dict:
    """text_mode='title': keep only the title as text. Same nodes, edges and splits as 'full' (the abstract is
    only used by the length filter), so it is a clean "scarce text" experiment. raw_dir reuses the raw files of
    another root (no second download)."""
    root = Path(root)
    raw = Path(raw_dir) if raw_dir else root / "raw"
    g = load_ogb(raw)
    ta_path = _find(raw, "titleabs.tsv*")
    if ta_path is None:
        raise SystemExit("titleabs.tsv.gz not found in raw/. Run python -m src.Ogbn.download first.")
    ta = load_titleabs(ta_path)

    n0 = len(g["year"])
    nodes = pd.DataFrame({"idx": np.arange(n0), "paper_id": g["paper_id"], "year": g["year"],
                          "label": g["label"], "split": g["split"]})
    nodes = nodes.merge(ta, on="paper_id", how="left")
    nodes["title"] = nodes["title"].fillna("").map(_clean)
    nodes["abstract"] = nodes["abstract"].fillna("").map(_clean)
    n_words = nodes["abstract"].str.split().str.len().fillna(0)
    keep = (n_words >= min_abstract_words) & (nodes["title"].str.len() > 0)
    stats = {"nodes_ogb": int(n0), "no_text_or_short": int((~keep).sum())}
    nodes = nodes[keep]
    stats["text_mode"] = text_mode

    if max_nodes and max_nodes < len(nodes):
        rng = np.random.default_rng(seed)
        picks = []
        for s in (0, 1, 2):
            ids = nodes.index[nodes["split"] == s].to_numpy()
            n_s = int(round(max_nodes * len(ids) / len(nodes)))
            picks.append(np.sort(rng.choice(ids, size=min(n_s, len(ids)), replace=False)))
        nodes = nodes.loc[np.concatenate(picks)].sort_values("idx")
        stats["subsampled_to"] = int(len(nodes))

    if text_mode == "title":
        nodes["abstract"] = ""
    old_idx = nodes["idx"].to_numpy()
    remap = np.full(n0, -1, dtype=np.int64)
    remap[old_idx] = np.arange(len(old_idx))
    e = g["edges"]
    e = e[(remap[e[:, 0]] >= 0) & (remap[e[:, 1]] >= 0)]
    e = np.stack([remap[e[:, 0]], remap[e[:, 1]]], axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0).astype(np.int32)

    nodes = nodes.reset_index(drop=True)
    nodes["idx"] = np.arange(len(nodes))
    names = g["label_names"] or {}

    def pretty(v: int) -> str:                      # "arxiv cs ai" -> "cs.AI"
        raw_name = names.get(int(v), str(v))
        parts = raw_name.split()
        return f"{parts[1]}.{parts[2].upper()}" if len(parts) == 3 and parts[0] == "arxiv" else raw_name

    nodes["label_name"] = nodes["label"].map(pretty)

    out = root / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    nodes[["idx", "paper_id", "year", "label", "label_name", "split", "title", "abstract"]].to_csv(
        out / "nodes.csv.gz", index=False)
    np.save(out / "edges.npy", e)
    stats.update({
        "nodes": int(len(nodes)), "edges": int(len(e)),
        "per_split": {n: int((nodes["split"] == s).sum()) for s, n in enumerate(["train", "val", "test"])},
        "years": {"min": int(nodes["year"].min()), "max": int(nodes["year"].max())},
        "classes": int(nodes["label"].nunique()),
        "edges_within_split": {n: int(((nodes["split"].to_numpy()[e[:, 0]] == s) &
                                       (nodes["split"].to_numpy()[e[:, 1]] == s)).sum())
                               for s, n in enumerate(["train", "val", "test"])},
    })
    (out / "meta.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--min-abstract-words", type=int, default=30)
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="subsample (stratified by split, seeded); citation edges become sparser")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-dir", default=None, help="reuse the raw files of another root, e.g. data/ogbn/raw")
    ap.add_argument("--text-mode", choices=["full", "title"], default="full")
    a = ap.parse_args()
    build(a.root, a.min_abstract_words, a.max_nodes, a.seed, a.raw_dir, a.text_mode)
