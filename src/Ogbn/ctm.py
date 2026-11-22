"""Neural topic model: ProdLDA / CombinedTM-style VAE with optional graph-propagated Transformer features.

Encoder input (per document), by variant:
    prodlda     : normalised BoW
    ctm         : normalised BoW  ++  embedding x
    ctm_knn/... : normalised BoW  ++  [x, A x, A^2 x]      (A = kNN / citation / union, see graphs.py)
Decoder: theta -> Linear -> BatchNorm -> + learnable log-unigram bias -> log-softmax over the vocabulary.
Anti-collapse measures from the earlier experiments are kept: decoder BN, dropout on theta, KL annealing,
Adam beta1=0.99, log-unigram bias, effective-beta term ranking.

Trained on TRAIN only; early stopping on the VAL loss; theta = softmax(mu) (deterministic).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from .evaluate import top_words


@dataclass
class Config:
    topics: int = 50
    hidden: int = 256
    dropout: float = 0.2
    theta_dropout: float = 0.2
    lr: float = 2e-3
    beta1: float = 0.99
    batch: int = 512
    epochs: int = 100
    patience: int = 8
    kl_weight: float = 1.0
    kl_anneal: int = 10          # epochs of linear KL warm-up
    decoder_bn: bool = True
    prior_alpha: float = 1.0     # Dirichlet alpha used in the Laplace approximation of the prior


class TopicVAE(nn.Module):
    def __init__(self, in_dim: int, vocab: int, cfg: Config, log_unigram: np.ndarray):
        super().__init__()
        K = cfg.topics
        self.K = K
        self.in_bn = nn.BatchNorm1d(in_dim)
        self.enc = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.mu = nn.Linear(cfg.hidden, K)
        self.logvar = nn.Linear(cfg.hidden, K)
        self.mu_bn = nn.BatchNorm1d(K, affine=False)
        self.lv_bn = nn.BatchNorm1d(K, affine=False)
        self.theta_drop = nn.Dropout(cfg.theta_dropout)
        self.dec = nn.Linear(K, vocab, bias=False)
        self.dec_bn = nn.BatchNorm1d(vocab, affine=False) if cfg.decoder_bn else nn.Identity()
        self.bias = nn.Parameter(torch.tensor(log_unigram, dtype=torch.float32))
        a = cfg.prior_alpha                                  # Laplace approx. of Dirichlet(alpha) (Srivastava & Sutton)
        pv = (1.0 / a) * (1 - 2.0 / K) + 1.0 / (K * a)
        self.register_buffer("prior_var", torch.tensor(float(pv)))

    def encode(self, x):
        h = self.enc(self.in_bn(x))
        return self.mu_bn(self.mu(h)), self.lv_bn(self.logvar(h))

    def decode_logp(self, theta):
        return F.log_softmax(self.dec_bn(self.dec(theta)) + self.bias, dim=-1)

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        theta = F.softmax(z, dim=-1)
        logp = self.decode_logp(self.theta_drop(theta) if sample else theta)
        return logp, mu, logvar

    def kl(self, mu, logvar):
        pv = self.prior_var
        return 0.5 * ((logvar.exp() + mu.pow(2)) / pv - 1.0 - logvar + torch.log(pv)).sum(1)

    @torch.no_grad()
    def topic_word(self) -> np.ndarray:
        """Effective beta: decode each topic as a one-hot theta in eval mode."""
        self.eval()
        eye = torch.eye(self.K, device=self.bias.device)
        return self.decode_logp(eye).exp().cpu().numpy()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_predict(bow: sp.csr_matrix, split: np.ndarray, feats: np.ndarray | None, use_bow: bool,
                cfg: Config, seed: int, top_n: int = 10, log=print):
    """bow: raw counts (N, V); feats: dense (N, D) graph/embedding features or None.
    Returns top_idx (K, top_n), theta {split: (N_s, K)}, info dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    N, V = bow.shape
    bow = bow.tocsr().astype(np.float32)
    bow_in = bow.copy()
    bow_in.data = np.log1p(bow_in.data)
    bow_in = normalize(bow_in, norm="l2")
    if not use_bow and feats is None:
        raise ValueError("model needs BoW input and/or features")
    in_dim = (V if use_bow else 0) + (0 if feats is None else feats.shape[1])

    tr, va = np.where(split == 0)[0], np.where(split == 1)[0]
    counts = np.asarray(bow[tr].sum(0)).ravel() + 1.0
    log_unigram = np.log(counts / counts.sum())
    model = TopicVAE(in_dim, V, cfg, log_unigram).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

    def batch(idx):
        parts = []
        if use_bow:
            parts.append(torch.from_numpy(bow_in[idx].toarray()))
        if feats is not None:
            parts.append(torch.from_numpy(feats[idx]))
        x = torch.cat(parts, 1).to(dev)
        tgt = torch.from_numpy(bow[idx].toarray()).to(dev)
        return x, tgt

    def run_epoch(idx, train: bool, kl_w: float) -> float:
        model.train(train)
        order = np.random.permutation(idx) if train else idx
        tot = 0.0
        for i0 in range(0, len(order), cfg.batch):
            b = order[i0:i0 + cfg.batch]
            if train and len(b) < 2:            # BatchNorm needs >1 row
                continue
            x, tgt = batch(b)
            with torch.set_grad_enabled(train):
                logp, mu, logvar = model(x, sample=train)
                rec = -(tgt * logp).sum(1)
                loss = (rec + kl_w * model.kl(mu, logvar)).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot += loss.item() * len(b)
        return tot / len(order)

    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(cfg.epochs):
        w = cfg.kl_weight * min(1.0, (ep + 1) / max(1, cfg.kl_anneal))
        tl = run_epoch(tr, True, w)
        vl = run_epoch(va, False, cfg.kl_weight)        # always full KL weight: comparable across epochs
        log(f"  epoch {ep + 1:3d}  train {tl:8.2f}  val {vl:8.2f}  (kl_w={w:.2f})")
        if ep + 1 >= cfg.kl_anneal:                      # only consider checkpoints after warm-up
            if vl < best - 1e-3:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"  early stop at epoch {ep + 1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    theta = {}
    with torch.no_grad():
        for s, name in enumerate(["train", "val", "test"]):
            idx = np.where(split == s)[0]
            out = []
            for i0 in range(0, len(idx), 2048):
                x, _ = batch(idx[i0:i0 + 2048])
                mu, _ = model.encode(x)
                out.append(F.softmax(mu, dim=-1).cpu().numpy())
            theta[name] = np.concatenate(out)
    info = {"epochs_run": ep + 1, "best_val_loss": float(best), "train_seconds": time.time() - t0,
            "device": str(dev)}
    return top_words(model.topic_word(), top_n), theta, info
