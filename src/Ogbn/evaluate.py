"""Metrics shared by ALL models (neural and baselines) so comparisons are like-for-like.

Every model must hand over:
    top_idx : (K, n) int  -- indices (into the shared vocabulary) of each topic's top-n words
    theta   : {split: (N_split, K)}  document-topic proportions (rows ~ sum to 1)

Metrics
    npmi        mean over topics of mean pairwise NPMI of the top-n words (document co-occurrence on the
                whole corpus' binary BoW; pairs that never co-occur score -1)
    diversity   |unique top words| / (K*n)           (Dieng et al. 2020)
    nmi         NMI between argmax topic and the arXiv category        (val/test)
    purity      cluster purity w.r.t. the arXiv category               (val/test)
    acc, f1     logistic regression on theta (train -> test), accuracy and macro-F1
    n_active    topics that are the argmax of >= 0.5% of the test documents
NPMI/diversity measure the topics; nmi/purity/acc/f1 measure the document representation.
Automatic coherence correlates imperfectly with human judgement: also inspect topics by eye.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def top_words(beta: np.ndarray, n: int = 10) -> np.ndarray:
    return np.argsort(-beta, axis=1)[:, :n]


def binary_csc(X: sp.spmatrix) -> sp.csc_matrix:
    return (X > 0).astype(np.float32).tocsc()


def npmi_per_topic(top_idx: np.ndarray, Xbin: sp.csc_matrix, eps: float = 1e-12) -> np.ndarray:
    N = Xbin.shape[0]
    df = np.asarray(Xbin.sum(0)).ravel() / N
    out = np.zeros(len(top_idx))
    for t, idx in enumerate(top_idx):
        sub = Xbin[:, idx]
        co = (sub.T @ sub).toarray() / N
        pi = np.maximum(df[idx], eps)
        pmi = np.log((co + eps) / (pi[:, None] * pi[None, :]))
        npmi = pmi / np.maximum(-np.log(co + eps), 1e-12)
        iu = np.triu_indices(len(idx), 1)
        out[t] = npmi[iu].mean()
    return out


def topic_diversity(top_idx: np.ndarray) -> float:
    return len(np.unique(top_idx)) / top_idx.size


def purity(pred: np.ndarray, y: np.ndarray) -> float:
    total = 0
    for c in np.unique(pred):
        total += np.bincount(y[pred == c]).max()
    return total / len(y)


def evaluate(top_idx: np.ndarray, theta: dict, labels: dict, Xbin: sp.csc_matrix,
             max_clf_train: int = 60000, seed: int = 0) -> dict:
    """labels: {split: (N_split,) int}. theta/labels keys: 'train','val','test'."""
    K = theta["test"].shape[1]
    m: dict = {}
    pt = npmi_per_topic(top_idx, Xbin)
    m["npmi"] = float(pt.mean())
    m["npmi_median"] = float(np.median(pt))
    m["diversity"] = topic_diversity(top_idx)
    for s in ("val", "test"):
        pred = theta[s].argmax(1)
        m[f"nmi_{s}"] = float(normalized_mutual_info_score(labels[s], pred))
        m[f"purity_{s}"] = float(purity(pred, labels[s]))
    pred_test = theta["test"].argmax(1)
    m["n_active"] = int((np.bincount(pred_test, minlength=K) >= 0.005 * len(pred_test)).sum())
    m["n_topics"] = int(K)

    rng = np.random.default_rng(seed)
    tr = theta["train"]
    ytr = labels["train"]
    if len(tr) > max_clf_train:
        sel = rng.choice(len(tr), max_clf_train, replace=False)
        tr, ytr = tr[sel], ytr[sel]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(np.log(tr + 1e-6), ytr)
    for s in ("val", "test"):
        p = clf.predict(np.log(theta[s] + 1e-6))
        m[f"acc_{s}"] = float((p == labels[s]).mean())
        m[f"f1_{s}"] = float(f1_score(labels[s], p, average="macro"))
    return m
