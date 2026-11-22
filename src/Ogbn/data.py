"""data.py
Loader shared by every later step."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SPLITS = ("train", "val", "test")


@dataclass
class Bundle:
    root: Path
    nodes: pd.DataFrame
    edges: np.ndarray            # (E, 2) int32, i cites j
    split: np.ndarray            # (N,) int8: 0/1/2
    labels: np.ndarray           # (N,) int64
    texts: list[str]             # title + ". " + abstract
    bow: sp.csr_matrix | None = None
    vocab: list[str] | None = None

    @property
    def fingerprint(self) -> str:
        """Identifies this exact node set/order: caches and results are keyed by it (no stale mixes)."""
        return hashlib.md5(self.nodes["paper_id"].to_numpy().astype("int64").tobytes()).hexdigest()[:8]

    @property
    def n(self) -> int:
        return len(self.nodes)

    def idx(self, s: int | str) -> np.ndarray:
        s = SPLITS.index(s) if isinstance(s, str) else s
        return np.where(self.split == s)[0]


def load(root: str | Path = "data/ogbn", with_bow: bool = True) -> Bundle:
    root = Path(root)
    nodes = pd.read_csv(root / "corpus" / "nodes.csv.gz", keep_default_na=False)
    edges = np.load(root / "corpus" / "edges.npy")
    b = Bundle(root=root, nodes=nodes, edges=edges,
               split=nodes["split"].to_numpy().astype(np.int8),
               labels=nodes["label"].to_numpy().astype(np.int64),
               texts=(nodes["title"] + ". " + nodes["abstract"]).tolist())
    if with_bow and (root / "features" / "bow.npz").exists():
        b.bow = sp.load_npz(root / "features" / "bow.npz").tocsr()
        b.vocab = json.loads((root / "features" / "vocab.json").read_text())
    return b


def load_embeddings(root: str | Path, name: str) -> np.ndarray:
    return np.load(Path(root) / "features" / f"emb_{name}.npy").astype(np.float32)
