"""Step 4 -- Transformer document embeddings of title+abstract (the "Transformer" requirement).

    python -m src.Ogbn.embeddings --root data/ogbn --model minilm            # fast, CPU-friendly
    python -m src.Ogbn.embeddings --root data/ogbn --model specter          # citation-informed (see note)
    python -m src.Ogbn.embeddings --root data/ogbn --model svd              # no Transformer; offline sanity baseline

Results: data/ogbn/features/emb_<model>.npy  (N x d, float32, L2-normalised).
Encoding is chunked and resumable (chunks in features/emb_<model>.parts/): Ctrl-C and rerun continues.

Note on SPECTER: it was trained with citation signal, so its embeddings already encode part of the graph.
Keep MiniLM as the main encoder and use SPECTER as an ablation when claiming "the graph adds information".
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import data

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "specter": "sentence-transformers/allenai-specter",
}


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_transformer(texts_title: list[str], texts_abs: list[str], key: str, out_dir: Path,
                       max_len: int = 256, batch: int = 64, chunk: int = 20000, fp: str = "") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELS[key], device=_device())
    model.max_seq_length = max_len
    sep = " [SEP] " if key == "specter" else ". "
    texts = [t + sep + a for t, a in zip(texts_title, texts_abs)]
    parts = out_dir / f"emb_{key}.{fp}.parts"       # fp: different corpus -> different chunks
    parts.mkdir(parents=True, exist_ok=True)
    pieces, t0 = [], time.time()
    for c, i0 in enumerate(range(0, len(texts), chunk)):
        f = parts / f"{c:05d}.npy"
        if not f.exists():
            e = model.encode(texts[i0:i0 + chunk], batch_size=batch, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
            np.save(f, e.astype(np.float32))
            print(f"[chunk {c}] {min(i0 + chunk, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s")
        pieces.append(np.load(f))
    return np.concatenate(pieces)


def encode_svd(texts: list[str], split: np.ndarray, dim: int = 128, seed: int = 0) -> np.ndarray:
    """TF-IDF + truncated SVD fit on the TRAIN split. Not a Transformer: used for tests / as a weak baseline."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    tr = np.where(split == 0)[0]
    tf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english")
    tf.fit([texts[i] for i in tr])
    X = tf.transform(texts)
    d = min(dim, X.shape[1] - 1)
    svd = TruncatedSVD(d, random_state=seed).fit(X[tr])
    return normalize(svd.transform(X)).astype(np.float32)


def run(root: str = "data/ogbn", model: str = "minilm", max_len: int = 256, batch: int = 64) -> Path:
    b = data.load(root, with_bow=False)
    out_dir = Path(root) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    if model == "svd":
        E = encode_svd(b.texts, b.split)
    else:
        E = encode_transformer(b.nodes["title"].tolist(), b.nodes["abstract"].tolist(), model,
                               out_dir, max_len, batch, fp=b.fingerprint)
    assert E.shape[0] == b.n, (E.shape, b.n)
    path = out_dir / f"emb_{model}.npy"
    np.save(path, E.astype(np.float32))
    print(f"[ok] {path}  shape={E.shape}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--model", default="minilm", choices=[*MODELS, "svd"])
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    run(a.root, a.model, a.max_len, a.batch)
