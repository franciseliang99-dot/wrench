"""Where generated text comes from (S1, S6, S8).

Live (default): AMD Token Factory, a hosted OpenAI-compatible endpoint whose models run on AMD GPUs (AMD Radeon Cloud).
Configured by environment variables, never by files in the repo:

    WRENCH_TF_KEY     the team's Token Factory API key
    WRENCH_TF_MODEL   model name (default Gemma-4-31B-IT)

Live (alternative, kept for a self-hosted run): the vLLM server on the MI300X, reached only through its reverse proxy over TLS with a self-signed certificate that is
the only one trusted (pinned). Configured by environment variables, never by files in the repo:

    WRENCH_GPU_URL    https://<ip>:<port>        (the IP changes per Droplet)
    WRENCH_GPU_KEY    bearer key checked by the proxy
    WRENCH_GPU_CERT   path to the proxy's certificate (its subjectAltName must contain that IP)
    WRENCH_GPU_MODEL  model name served by vLLM

Development only: WRENCH_GENERATOR=stub uses the CPU stub from qa.py and reports itself as "cpu-stub".

Replay: artifacts/recorded.json holds raw model outputs captured from the live generator by scripts/record_answers.py, keyed by the
normalised question. A replayed output still goes through the same citation check as a live one.
"""
from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import qa
from .paths import REPO

RECORDED = REPO / "artifacts" / "recorded.json"
MAX_TOKENS = 600  # S8: fixed cap per call
TIMEOUT = 60.0
TF_TIMEOUT = 120.0  # Gemma-4-31B-IT on Token Factory took 15-34 s for a 218-token answer (2026-10-17)


class GenerationOffline(Exception):
    """The live generator could not produce an answer. The message never contains the key."""


@dataclass
class Live:
    name: str  # "amd-tokenfactory", "mi300x" or "cpu-stub"

    def __call__(self, messages: list[dict]) -> str:
        raise NotImplementedError


class Stub(Live):
    def __init__(self):
        super().__init__("cpu-stub")

    def __call__(self, messages: list[dict]) -> str:
        return qa.stub_generate(messages)


class MI300X(Live):
    def __init__(self, url: str, key: str, cert: str, model: str):
        super().__init__("mi300x")
        if not url.startswith("https://"):
            raise ValueError("WRENCH_GPU_URL must be https:// (the proxy terminates TLS)")
        ctx = ssl.create_default_context(cafile=cert)  # trusts this certificate and nothing else; missing file fails here
        self._client = httpx.Client(base_url=url.rstrip("/"), verify=ctx, timeout=TIMEOUT,
                                    headers={"Authorization": f"Bearer {key}"})
        self._model = model

    def __call__(self, messages: list[dict]) -> str:
        body = {"model": self._model, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0}
        try:
            r = self._client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            raise GenerationOffline(type(e).__name__) from None


class TokenFactory(Live):
    BASE = "https://developer.amd.com.cn/radeon/api/v1"
    DEFAULT_MODEL = "Gemma-4-31B-IT"

    def __init__(self, key: str, model: str = DEFAULT_MODEL, base: str = BASE, transport: httpx.BaseTransport | None = None):
        super().__init__("amd-tokenfactory")
        self._client = httpx.Client(base_url=base.rstrip("/"), timeout=TF_TIMEOUT, transport=transport,
                                    headers={"Authorization": f"Bearer {key}"})
        self._model = model

    def __call__(self, messages: list[dict]) -> str:
        body = {"model": self._model, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0}
        try:
            r = self._client.post("/chat/completions", json=body)
            r.raise_for_status()  # 429 (shared per-key limit) becomes offline: the API then falls back to a recording
            out = r.json()["choices"][0]["message"]["content"]
            if not isinstance(out, str):
                raise TypeError
            return out
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            raise GenerationOffline(type(e).__name__) from None


def live_from_env(env: dict | None = None) -> Live | None:
    env = os.environ if env is None else env
    if env.get("WRENCH_GENERATOR") == "stub":
        return Stub()
    if env.get("WRENCH_TF_KEY"):
        return TokenFactory(env["WRENCH_TF_KEY"], env.get("WRENCH_TF_MODEL") or TokenFactory.DEFAULT_MODEL)
    url = env.get("WRENCH_GPU_URL")
    if not url:
        return None
    missing = [k for k in ("WRENCH_GPU_KEY", "WRENCH_GPU_CERT", "WRENCH_GPU_MODEL") if not env.get(k)]
    if missing:
        raise ValueError(f"WRENCH_GPU_URL is set but {', '.join(missing)} is not")
    return MI300X(url, env["WRENCH_GPU_KEY"], env["WRENCH_GPU_CERT"], env["WRENCH_GPU_MODEL"])


def norm_question(q: str) -> str:
    return " ".join(q.lower().split())


def load_recorded(path: Path = RECORDED) -> dict:
    """{"machine": ..., "model": ..., "answers": {normalised question: {"output": raw, "recorded_at": iso}}}; {} if absent."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data.get("answers"), dict) else {}
