"""Load deepset/prompt-injections (train/test parquet) and the local challenge set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def load_deepset() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(snapshot_download("deepset/prompt-injections", repo_type="dataset",
                                  allow_patterns="data/*.parquet", local_dir=ROOT / "data" / "deepset"))
    files = sorted((path / "data").glob("*.parquet"))
    train = pd.concat(pd.read_parquet(f) for f in files if f.name.startswith("train"))
    test = pd.concat(pd.read_parquet(f) for f in files if f.name.startswith("test"))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_challenge() -> pd.DataFrame:
    rows = [json.loads(line) for line in (ROOT / "evals" / "challenge.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)
