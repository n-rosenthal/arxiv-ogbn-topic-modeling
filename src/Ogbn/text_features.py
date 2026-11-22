"""Step 3 -- bag-of-words over title+abstract. Vocabulary is fit on the TRAIN split only (no leakage).

    python -m src.Ogbn.text_features --root data/ogbn --min-df 20 --max-df 0.3 --max-features 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from . import data

# Generic abstract-writing words that carry no topic signal (keep short; max_df removes the rest).
EXTRA_STOP = {"paper", "propose", "proposed", "present", "show", "shows", "result", "results",
              "approach", "method", "methods", "using", "use", "used", "based", "new", "novel",
              "study", "work", "also", "however", "can", "may", "one", "two", "al", "et"}
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b"


def make_vectorizer(min_df: int, max_df: float, max_features: int | None, ngram: int = 1,
                    vocabulary: list[str] | None = None) -> CountVectorizer:
    return CountVectorizer(lowercase=True, token_pattern=TOKEN,
                           stop_words=sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP),
                           min_df=min_df, max_df=max_df, max_features=max_features,
                           ngram_range=(1, ngram), dtype=np.float32, vocabulary=vocabulary)


def run(root: str = "data/ogbn", min_df: int = 20, max_df: float = 0.3,
        max_features: int | None = 5000, ngram: int = 1) -> dict:
    b = data.load(root, with_bow=False)
    train_texts = [b.texts[i] for i in b.idx("train")]
    vec = make_vectorizer(min_df, max_df, max_features, ngram)
    vec.fit(train_texts)
    X = vec.transform(b.texts).tocsr()
    vocab = vec.get_feature_names_out().tolist()
    out = Path(root) / "features"
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "bow.npz", X)
    (out / "vocab.json").write_text(json.dumps(vocab))
    lens = np.asarray(X.sum(1)).ravel()
    info = {"docs": X.shape[0], "vocab": X.shape[1], "empty_docs": int((lens == 0).sum()),
            "median_tokens": float(np.median(lens)), "nnz_per_doc": float(X.nnz / X.shape[0])}
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-df", type=float, default=0.3)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--ngram", type=int, default=1)
    a = ap.parse_args()
    run(a.root, a.min_df, a.max_df, a.max_features, a.ngram)
