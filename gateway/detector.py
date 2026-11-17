"""Prompt-injection detectors and the Guard that combines them.

Detectors return a probability that a text is an injection attack (0..1).
The Guard runs cheap rules first, then a model detector over overlapping
chunks (long documents are split so an attack hidden on page 5 is still seen),
and blocks when any chunk scores at or above the threshold.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Verdict:
    blocked: bool
    score: float
    detector: str
    reason: str
    latency_ms: float


class Detector(Protocol):
    name: str

    def score(self, text: str) -> float: ...


# --- Rules: high-precision patterns, English + German -------------------------

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,20}\b(instructions?|prompts?|rules|directions)\b", re.I | re.S), "override instructions"),
    (re.compile(r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system prompt|hidden prompt|initial instructions|your instructions)\b", re.I | re.S), "prompt extraction"),
    (re.compile(r"\b(developer mode|jailbreak|DAN mode|do anything now)\b", re.I), "jailbreak keyword"),
    (re.compile(r"\byou are no longer\b|\bfrom now on,? you (are|will)\b", re.I), "role reassignment"),
    (re.compile(r"\b(ignoriere|vergiss|missachte)\b.{0,40}\b(anweisungen|befehle|regeln|alles)\b", re.I | re.S), "override instructions (de)"),
]


class RuleDetector:
    name = "rules"

    def match(self, text: str) -> str | None:
        for pattern, reason in RULES:
            if pattern.search(text):
                return reason
        return None

    def score(self, text: str) -> float:
        return 1.0 if self.match(text) else 0.0


# --- TF-IDF + logistic regression ---------------------------------------------

class TfidfDetector:
    name = "tfidf"

    def __init__(self, path: str | Path):
        import joblib

        self.pipeline = joblib.load(path)

    def score(self, text: str) -> float:
        return float(self.pipeline.predict_proba([text])[0][1])


# --- LoRA fine-tuned Qwen2.5-1.5B (optional, needs torch/transformers/peft) ----

class LoraDetector:
    name = "lora"

    def __init__(self, adapter: str, base: str = "Qwen/Qwen2.5-1.5B-Instruct", max_length: int = 128):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
        model.config.pad_token_id = self.tok.pad_token_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PeftModel.from_pretrained(model, adapter).to(self.device).eval()
        self.max_length = max_length

    def score(self, text: str) -> float:
        inputs = self.tok(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        return float(self._torch.softmax(logits, dim=-1)[0, 1])


# --- Guard --------------------------------------------------------------------

def chunks(text: str, size: int = 600, overlap: int = 150) -> list[str]:
    """Overlapping character windows, so long inputs are fully covered."""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text) - overlap, step)]


class Guard:
    def __init__(self, model: Detector | None = None, threshold: float = 0.5, use_rules: bool = True):
        self.rules = RuleDetector() if use_rules else None
        self.model = model
        self.threshold = threshold

    @property
    def name(self) -> str:
        parts = (["rules"] if self.rules else []) + ([self.model.name] if self.model else [])
        return "+".join(parts) or "none"

    def check(self, text: str) -> Verdict:
        start = time.perf_counter()
        if self.rules and (reason := self.rules.match(text)):
            return Verdict(True, 1.0, "rules", reason, _ms(start))
        if self.model is None:
            return Verdict(False, 0.0, self.name, "no rule matched", _ms(start))
        score = max(self.model.score(c) for c in chunks(text))
        blocked = score >= self.threshold
        reason = f"{self.model.name} score {score:.2f} {'>=' if blocked else '<'} {self.threshold}"
        return Verdict(blocked, round(score, 4), self.model.name, reason, _ms(start))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def build_guard(detector: str = "tfidf", threshold: float = 0.5, tfidf_path: str = "models/tfidf.joblib",
                lora_adapter: str = "", lora_base: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Guard:
    if detector == "rules":
        return Guard(None, threshold)
    if detector == "tfidf":
        return Guard(TfidfDetector(tfidf_path), threshold)
    if detector == "lora":
        if not lora_adapter:
            raise ValueError("GUARD_DETECTOR=lora needs LORA_ADAPTER (Hugging Face repo id or local path)")
        return Guard(LoraDetector(lora_adapter, lora_base), threshold)
    raise ValueError(f"unknown detector: {detector}")
