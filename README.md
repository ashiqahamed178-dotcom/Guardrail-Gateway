# Guardrail Gateway

An OpenAI-compatible proxy that checks every message for prompt-injection attacks before it reaches your LLM. Point your app at the gateway instead of the provider (one URL change); clean requests are forwarded unchanged, attacks are blocked with an OpenAI-style error and logged.

```
your app ──► Guardrail Gateway ──► OpenAI / Gemini / Groq / Ollama
               │  rules (regex, EN + DE)
               │  model detector: TF-IDF (default) or LoRA-tuned Qwen2.5-1.5B
               └► SQLite log + dashboard (/), stats (/stats)
```

## Why

Prompt injection is the top risk in the OWASP Top 10 for LLM applications. Attacks arrive directly from users and indirectly through documents, web pages and tool results that the model reads. This gateway checks **user messages and tool outputs** (indirect injection), splits long inputs into overlapping chunks so an attack hidden at the end of a document is still seen, and records every decision with a reason.

## Results

Two test sets:
- **deepset test**: the public [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) test split (116 examples, 60 attacks, English and German).
- **challenge set** (`evals/challenge.jsonl`): 20 hand-written examples, 8 attacks (5 hidden inside documents or tool output) and 12 harmless "hard negatives" that use attack-like words ("instructions", "forget my earlier question", "act as a tutor").

| Config | deepset F1 | deepset recall | deepset false alarms | challenge attacks caught | challenge false alarms | p50 latency |
|---|---|---|---|---|---|---|
| rules only | 0.154 | 0.083 | 0 | 4 / 8 | 0 / 12 | 0.02 ms |
| TF-IDF only | 0.849 | 0.750 | 1 | 8 / 8 | **8 / 12** | 0.7 ms |
| rules + TF-IDF (default) | 0.849 | 0.750 | 1 | 8 / 8 | 8 / 12 | 0.7 ms |
| LoRA Qwen2.5-1.5B ([training repo](https://github.com/ashiqahamed178-dotcom/Prompt-Injection-Lora)) | 0.983 | 0.967 | 0 | to measure | to measure | to measure |

Full numbers, missed attacks and false-alarm examples: `evals/results.json` (regenerate with `python scripts/evaluate.py`).

### What the numbers show

- **Rules are precise but blind.** No false alarms, but they catch only 8% of the deepset attacks and miss every indirect attack that does not use the words "ignore previous instructions".
- **TF-IDF catches every challenge attack but blocks 8 of 12 harmless requests**, including "Write a short story where a robot learns to paint" and "Translate this sentence to German". The deepset training data labels task switches ("write a story", "translate to...") as attacks, so a bag-of-words model learns that ordinary task words mean "attack". On the deepset test set this is invisible (1 false alarm); on realistic traffic it would block a large share of normal use.
- That over-blocking is the main reason to use the LoRA detector, which reads context rather than keywords. Measuring its false-alarm rate on the challenge set is the next step.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt pytest
python scripts/train_tfidf.py                     # downloads the dataset, trains models/tfidf.joblib
pytest -q                                         # 9 tests
python scripts/evaluate.py --check                # results table + CI eval gate
UPSTREAM_BASE_URL=https://api.groq.com/openai/v1/ UPSTREAM_API_KEY=... uvicorn gateway.main:app
```

Then use any OpenAI client with `base_url="http://localhost:8000/v1"`. Open `http://localhost:8000/` for the dashboard.

Docker: `docker build -t guardrail-gateway . && docker run -p 8000:8000 -e UPSTREAM_BASE_URL=... -e UPSTREAM_API_KEY=... guardrail-gateway`

### API

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | Checks user and tool messages; blocked → `400 {"error": {"type": "prompt_injection_blocked", ...}}`, clean → forwarded to the upstream |
| `POST /v1/check` | `{"text": "..."}` → verdict (blocked, score, detector, reason, latency) |
| `GET /stats` | Counts, block reasons, p50/p95 latency, last 20 decisions |
| `GET /` | Dashboard |
| `GET /health` | Detector, threshold, upstream configured |

### Configuration

| Variable | Default | |
|---|---|---|
| `GUARD_DETECTOR` | `tfidf` | `rules`, `tfidf` or `lora` (rules always run first) |
| `GUARD_THRESHOLD` | `0.5` | Block when the model score is at or above this |
| `UPSTREAM_BASE_URL` / `UPSTREAM_API_KEY` | none | Any OpenAI-compatible provider |
| `LORA_ADAPTER` | none | Hugging Face repo id or local path of the LoRA adapter (needs `requirements-lora.txt`) |
| `LOG_TEXT` | `0` | `1` stores the first 200 characters; by default only a SHA-256 prefix is logged |

## Design decisions

- **Rules first, model second.** Rules cost microseconds and never need a GPU; the model only runs when no rule matches.
- **Tool output is checked, system prompts are not.** The system prompt is written by the app owner; tool results and retrieved documents come from outside and are the main indirect-injection path.
- **Chunking.** Long inputs are scored in 600-character windows with 150 characters of overlap, and the highest score decides.
- **Privacy by default.** The log stores a hash, not the text.
- **Eval gate in CI.** `scripts/evaluate.py --check` fails the build if the default detector catches fewer attacks or raises more false alarms than the measured baseline.

## Limits

- Small test sets (116 + 20); the challenge set is hand-written and labelled by the author.
- Streaming requests are rejected for now.
- The block/pass decision uses a single threshold; there is no "flag for review" middle band yet.
- The LoRA detector needs about 6 GB of RAM on CPU and adds latency; its numbers on the challenge set are not measured yet.
