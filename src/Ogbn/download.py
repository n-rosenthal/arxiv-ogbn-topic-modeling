"""Step 1 -- download ogbn-arxiv (graph, years, labels, splits) and the raw title/abstract file.

    python -m src.Ogbn.download --root data/ogbn

No dependency on the `ogb` package: plain HTTPS download + unzip.
Layout produced:
    data/ogbn/raw/arxiv.zip, raw/titleabs.tsv.gz, raw/arxiv/{raw,split,mapping}/...
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "arxiv.zip": "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    "titleabs.tsv.gz": "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz",
}


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already there ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[get ] {url} (attempt {attempt})")
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, next_report = 0, 0.1
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total and done / total >= next_report:
                        print(f"       {done / 1e6:.0f}/{total / 1e6:.0f} MB")
                        next_report += 0.1
            part.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] {type(e).__name__}: {e}")
    raise SystemExit(
        f"Could not download {url}: {last_err}\n"
        f"Download it manually in a browser and save it as {dest}."
    )


def run(root: str | Path = "data/ogbn") -> Path:
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        fetch(url, raw / name)
    target = raw / "arxiv"
    if not target.exists():
        print("[unzip] arxiv.zip")
        with zipfile.ZipFile(raw / "arxiv.zip") as z:
            z.extractall(raw)
    print(f"[ok] raw data in {raw}")
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ogbn")
    run(ap.parse_args().root)
    sys.exit(0)
