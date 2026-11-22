"""Non-neural / non-graph baselines trained on the TRAIN split only; val/test are folded in.

    lda       sklearn online LDA on BoW counts
    nmf       sklearn NMF on tf-idf
    bertopic  BERTopic on the same Transformer embeddings (needs: pip install bertopic umap-learn hdbscan)
Each returns (top_idx, theta) as expected by evaluate.evaluate().
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer

from .evaluate import top_words


def _rownorm(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64) + eps
    return M / M.sum(1, keepdims=True)


def run_lda(bow: sp.csr_matrix, split: np.ndarray, K: int, seed: int, top_n: int = 10, max_iter: int = 20):
    tr = np.where(split == 0)[0]
    lda = LatentDirichletAllocation(n_components=K, learning_method="online", batch_size=2048,
                                    max_iter=max_iter, random_state=seed, n_jobs=-1, evaluate_every=-1)
    lda.fit(bow[tr])
    theta = {n: _rownorm(lda.transform(bow[split == s])) for s, n in enumerate(["train", "val", "test"])}
    return top_words(lda.components_, top_n), theta


def run_nmf(bow: sp.csr_matrix, split: np.ndarray, K: int, seed: int, top_n: int = 10, max_iter: int = 300):
    tr = np.where(split == 0)[0]
    tfidf = TfidfTransformer(sublinear_tf=True).fit(bow[tr])
    X = tfidf.transform(bow)
    nmf = NMF(n_components=K, init="nndsvda", max_iter=max_iter, random_state=seed)
    nmf.fit(X[tr])
    theta = {n: _rownorm(nmf.transform(X[split == s])) for s, n in enumerate(["train", "val", "test"])}
    return top_words(nmf.components_, top_n), theta


def run_bertopic(texts: list[str], emb: np.ndarray, split: np.ndarray, vocab: list[str], K: int,
                 seed: int, top_n: int = 10):
    """Hard assignments (one-hot theta); outlier docs (-1) get an all-zero row (then uniform).
    The number of topics found can differ from K: it is reported as n_topics in the results."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    from .text_features import make_vectorizer
    vec = make_vectorizer(1, 1.0, None, vocabulary=vocab)
    tr = np.where(split == 0)[0]
    tm = BERTopic(embedding_model=None, vectorizer_model=vec, nr_topics=K, top_n_words=top_n,
                  umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
                  hdbscan_model=HDBSCAN(min_cluster_size=max(30, len(tr) // 1000), prediction_data=True),
                  calculate_probabilities=False, verbose=False)
    topics_tr, _ = tm.fit_transform([texts[i] for i in tr], embeddings=emb[tr])
    ids = sorted(t for t in set(tm.get_topics()) if t != -1)
    col = {t: j for j, t in enumerate(ids)}
    w2i = {w: i for i, w in enumerate(vocab)}
    top_idx = np.array([[w2i[w] for w, _ in tm.get_topic(t)[:top_n] if w in w2i] +
                        [0] * max(0, top_n - len(tm.get_topic(t)[:top_n])) for t in ids])

    def onehot(topics):
        T = np.zeros((len(topics), len(ids)))
        for r, t in enumerate(topics):
            if t in col:
                T[r, col[t]] = 1.0
        T[T.sum(1) == 0] = 1.0 / len(ids)
        return T

    theta = {"train": onehot(topics_tr)}
    for s, n in ((1, "val"), (2, "test")):
        ix = np.where(split == s)[0]
        t, _ = tm.transform([texts[i] for i in ix], embeddings=emb[ix])
        theta[n] = onehot(t)
    return top_idx, theta
