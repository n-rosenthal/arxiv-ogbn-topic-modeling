"""Step 5 -- graphs (kNN on embeddings, citations) with a leakage-free temporal protocol.

Protocol ("inductive by split"): a node of split s may only have neighbours in the splits <= s
    train rows -> neighbours among train
    val   rows -> neighbours among train + val
    test  rows -> neighbours among train + val + test
(citations point to older papers, so a train paper never "sees" a future citer.)
`neighbour rank <= row rank` implements this in one comparison.

Neighbour features are then precomputed (decoupled propagation, SIGN-style):
    F = [X, A X, A^2 X],  A = row-normalised (mean) adjacency.
The encoder's first layer is a learned linear map over F, i.e. a learned graph convolution with fixed
propagation. This makes GNN training as cheap as an MLP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def citation_pairs(edges: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected neighbour pairs (row, neighbour) allowed by the protocol."""
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    a, b = np.concatenate([u, v]), np.concatenate([v, u])
    ok = split[b] <= split[a]
    return a[ok], b[ok]


def _knn_block(E: np.ndarray, rows: np.ndarray, pool: np.ndarray, k: int, chunk: int = 512):
    n = E.shape[0]
    P = E[pool]
    pos = np.full(n, -1, dtype=np.int64)
    pos[pool] = np.arange(len(pool))
    kk = min(k, len(pool) - 1)
    out_r, out_n = [], []
    for i0 in range(0, len(rows), chunk):
        q = rows[i0:i0 + chunk]
        S = E[q] @ P.T
        sp_ = pos[q]
        has = np.where(sp_ >= 0)[0]
        S[has, sp_[has]] = -np.inf                      # no self-loops
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        out_r.append(np.repeat(q, kk))
        out_n.append(pool[idx].ravel())
    return np.concatenate(out_r), np.concatenate(out_n)


def knn_pairs(E: np.ndarray, split: np.ndarray, k: int, cache: Path | None = None):
    if cache is not None and cache.exists():
        z = np.load(cache)
        return z["rows"], z["nbrs"]
    rows, nbrs = [], []
    for s in (0, 1, 2):
        r = np.where(split == s)[0]
        pool = np.where(split <= s)[0]
        if len(r) == 0:
            continue
        a, b = _knn_block(E, r, pool, k)
        rows.append(a)
        nbrs.append(b)
    rows, nbrs = np.concatenate(rows), np.concatenate(nbrs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, rows=rows, nbrs=nbrs)
    return rows, nbrs


def to_mean_adjacency(n: int, rows: np.ndarray, nbrs: np.ndarray) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, nbrs)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0                                     # binary, de-duplicated
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv.astype(np.float32)) @ A


def random_pairs(split: np.ndarray, k: int, seed: int = 0):
    """Control graph: each row gets k RANDOM neighbours from its allowed pool (same protocol/degree as kNN)."""
    rng = np.random.default_rng(seed)
    rows, nbrs = [], []
    for s in (0, 1, 2):
        r = np.where(split == s)[0]
        pool = np.where(split <= s)[0]
        if len(r) == 0:
            continue
        a = np.repeat(r, k)
        b = pool[rng.integers(0, len(pool), size=len(a))]
        ok = a != b
        rows.append(a[ok])
        nbrs.append(b[ok])
    return np.concatenate(rows), np.concatenate(nbrs)


def build_adjacency(kind: str, E: np.ndarray, edges: np.ndarray, split: np.ndarray,
                    k: int = 10, cache_dir: Path | None = None, emb_name: str = "") -> sp.csr_matrix | None:
    """kind in {'none','knn','cite','both','rand'} (rand = random-neighbour control)."""
    if kind == "none":
        return None
    n = len(split)
    rows, nbrs = [], []
    if kind == "rand":
        a, b = random_pairs(split, k)
        return to_mean_adjacency(n, a, b)
    if kind in ("cite", "both"):
        a, b = citation_pairs(edges, split)
        rows.append(a)
        nbrs.append(b)
    if kind in ("knn", "both"):
        cache = None if cache_dir is None else cache_dir / f"knn_{emb_name}_k{k}.npz"
        a, b = knn_pairs(E, split, k, cache)
        rows.append(a)
        nbrs.append(b)
    return to_mean_adjacency(n, np.concatenate(rows), np.concatenate(nbrs))


def propagate(X: np.ndarray, A: sp.csr_matrix | None, hops: int = 1) -> np.ndarray:
    """[X, AX, A^2 X, ...] (float32). With A=None returns X."""
    if A is None:
        return X.astype(np.float32)
    feats, H = [X], X
    for _ in range(hops):
        H = A @ H
        feats.append(H)
    return np.hstack(feats).astype(np.float32)


def graph_stats(A: sp.csr_matrix | None, split: np.ndarray) -> dict:
    if A is None:
        return {}
    deg = np.diff(A.indptr)
    return {name: {"mean_degree": float(deg[split == s].mean()),
                   "isolated_frac": float((deg[split == s] == 0).mean())}
            for s, name in enumerate(["train", "val", "test"]) if (split == s).any()}
