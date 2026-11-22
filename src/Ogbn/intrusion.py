"""Human word-intrusion test (Chang et al., 2009), blind and paired, to complement NPMI and the weak labels.

    python -m src.Ogbn.intrusion make  --root data/ogbn --emb minilm --topics 50 --models ctm ctm_both --seed 0 --n 25
    # -> <results>/intrusion_form.org (answer sheet, blind) and intrusion_key.json (do NOT open before answering)
    # fill the "resposta" column with the number (1-6) of the word that does not belong; leave blank if unsure
    python -m src.Ogbn.intrusion score --root data/ogbn --emb minilm --topics 50

Each item: the 5 top words of a topic plus one intruder (a top word of another random topic of the same model that is
not among the first 10 words of the evaluated topic), shuffled. Items of all models are mixed and shuffled so the
rater cannot tell the model. Score = share of items where the intruder is found; the CI is a bootstrap over topics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate


def make_items(top_words: dict[str, list[list[str]]], n: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    items = []
    for model, tw in top_words.items():
        K = len(tw)
        chosen = rng.choice(K, size=min(n, K), replace=False)
        for t in chosen:
            pool = [w for o in range(K) if o != t for w in tw[o][:5] if w not in tw[t][:10]]
            intruder = str(rng.choice(pool))
            words = list(tw[t][:5]) + [intruder]
            order = rng.permutation(6)
            shown = [words[i] for i in order]
            items.append({"model": model, "topic": int(t), "words": shown, "intruder_pos": int(shown.index(intruder)) + 1})
    perm = rng.permutation(len(items))
    items = [items[i] for i in perm]
    form = [{"id": i + 1, "words": it["words"]} for i, it in enumerate(items)]
    key = [{"id": i + 1, "model": it["model"], "topic": it["topic"], "intruder_pos": it["intruder_pos"]}
           for i, it in enumerate(items)]
    return form, key


def form_org(form: list[dict]) -> str:
    out = ["| id | 1 | 2 | 3 | 4 | 5 | 6 | resposta |", "|----+---+---+---+---+---+---+----------|"]
    for f in form:
        out.append(f"| {f['id']} | " + " | ".join(f["words"]) + " |  |")
    return "\n".join(out) + "\n"


def parse_answers(text: str) -> dict[int, int | None]:
    ans = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0].isdigit():
            ans[int(cells[0])] = int(cells[7]) if cells[7].isdigit() else None
    return ans


def score(key: list[dict], ans: dict[int, int | None], n_boot: int = 5000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = {}
    for k in key:
        a = ans.get(k["id"])
        if a is None:
            continue
        by.setdefault(k["model"], {})[k["topic"]] = float(a == k["intruder_pos"])
    rows = []
    for m, d in by.items():
        v = np.array(list(d.values()))
        boot = rng.choice(v, size=(n_boot, len(v))).mean(axis=1)
        rows.append({"modelo": aggregate.NAMES.get(m, m), "itens": len(v), "acerto": f"{v.mean():.3f}",
                     "IC95% (bootstrap)": f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"})
    return rows


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--root", default="data/ogbn")
    ap.add_argument("--emb", default="minilm")
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["ctm", "ctm_both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    d = aggregate.results_dir(a.root, a.emb)
    if a.cmd == "make":
        tw = {m: json.loads((d / f"{m}_K{a.topics}_s{a.seed}.json").read_text())["top_words"] for m in a.models}
        form, key = make_items(tw, a.n, a.seed)
        (d / "intrusion_form.org").write_text(form_org(form))
        (d / "intrusion_key.json").write_text(json.dumps(key))
        print(f"{len(form)} items -> {d / 'intrusion_form.org'}  (do not open intrusion_key.json)")
    else:
        key = json.loads((d / "intrusion_key.json").read_text())
        ans = parse_answers((d / "intrusion_form.org").read_text())
        t = pd.DataFrame(score(key, ans))
        (d / "intrusion_result.org").write_text(aggregate.to_org(t))
        print(t.to_string(index=False))
