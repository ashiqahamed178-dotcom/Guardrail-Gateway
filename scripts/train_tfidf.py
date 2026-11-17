"""Train the TF-IDF + logistic regression detector on the deepset train split."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load_deepset  # noqa: E402


def main() -> None:
    train, _ = load_deepset()
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipe.fit(train["text"], train["label"])
    out = ROOT / "models" / "tfidf.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out)
    print(f"trained on {len(train)} examples -> {out}")


if __name__ == "__main__":
    main()
