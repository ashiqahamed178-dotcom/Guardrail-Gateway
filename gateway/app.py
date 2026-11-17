"""OpenAI-compatible guard proxy.

Point any OpenAI-compatible client at this server instead of the provider.
Every user and tool message is checked; attacks are blocked with an
OpenAI-style error, clean requests are forwarded unchanged to the upstream.
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .detector import Guard, build_guard
from .store import EventStore

CHECKED_ROLES = {"user", "tool", "function"}  # tool output = indirect injection risk


class CheckRequest(BaseModel):
    text: str


def message_text(content) -> str:
    """OpenAI content can be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def create_app(guard: Guard | None = None, store: EventStore | None = None,
               upstream: httpx.AsyncClient | None = None) -> FastAPI:
    guard = guard or build_guard(
        detector=os.getenv("GUARD_DETECTOR", "tfidf"),
        threshold=float(os.getenv("GUARD_THRESHOLD", "0.5")),
        tfidf_path=os.getenv("TFIDF_PATH", "models/tfidf.joblib"),
        lora_adapter=os.getenv("LORA_ADAPTER", ""),
    )
    store = store or EventStore(os.getenv("GUARD_DB", "guard.db"), os.getenv("LOG_TEXT", "0") == "1")
    base_url = os.getenv("UPSTREAM_BASE_URL", "")
    upstream = upstream or (httpx.AsyncClient(base_url=base_url, timeout=60) if base_url else None)
    api_key = os.getenv("UPSTREAM_API_KEY", "")

    app = FastAPI(title="Guardrail Gateway")

    @app.get("/health")
    def health():
        return {"status": "ok", "detector": guard.name, "threshold": guard.threshold,
                "upstream": bool(upstream)}

    @app.post("/v1/check")
    def check(req: CheckRequest):
        v = guard.check(req.text)
        store.add("/v1/check", "user", v, req.text)
        return v.__dict__

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):
            return _error(400, "invalid_request_error", "Streaming is not supported by the gateway yet.")
        for i, msg in enumerate(body.get("messages", [])):
            role = msg.get("role", "")
            if role not in CHECKED_ROLES:
                continue
            text = message_text(msg.get("content"))
            if not text.strip():
                continue
            v = guard.check(text)
            store.add("/v1/chat/completions", role, v, text)
            if v.blocked:
                return _error(400, "prompt_injection_blocked",
                              f"Message {i} ({role}) was blocked: {v.reason}.",
                              extra={"detector": v.detector, "score": v.score})
        if upstream is None:
            return _error(503, "upstream_not_configured", "Set UPSTREAM_BASE_URL to forward clean requests.")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = await upstream.post("chat/completions", json=body, headers=headers)
        return JSONResponse(resp.json(), status_code=resp.status_code, headers={"x-guard": "passed"})

    @app.get("/stats")
    def stats():
        return store.stats()

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        s = store.stats()
        rows = "".join(
            f"<tr><td>{datetime.fromtimestamp(e['ts'], timezone.utc):%Y-%m-%d %H:%M:%S}</td>"
            f"<td>{html.escape(e['role'])}</td><td>{'BLOCKED' if e['blocked'] else 'passed'}</td>"
            f"<td>{e['score']:.2f}</td><td>{html.escape(e['reason'])}</td><td>{e['latency_ms']}</td>"
            f"<td><code>{html.escape(e['text_ref'])}</code></td></tr>"
            for e in s["recent"])
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Guardrail Gateway</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left;font-size:14px}}.k{{display:inline-block;margin-right:24px}}
.k b{{font-size:24px;display:block}}</style></head><body>
<h1>Guardrail Gateway</h1><p>Detector: <b>{html.escape(guard.name)}</b>, threshold {guard.threshold}</p>
<div class="k"><b>{s['checked']}</b>messages checked</div><div class="k"><b>{s['blocked']}</b>blocked</div>
<div class="k"><b>{s['latency_ms_p50']}</b>p50 ms</div><div class="k"><b>{s['latency_ms_p95']}</b>p95 ms</div>
<h2>Recent</h2><table><tr><th>Time (UTC)</th><th>Role</th><th>Result</th><th>Score</th><th>Reason</th><th>ms</th><th>Text</th></tr>
{rows}</table></body></html>"""

    return app


def _error(status: int, kind: str, message: str, extra: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message, **(extra or {})}}, status_code=status)
