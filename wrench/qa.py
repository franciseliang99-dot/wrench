"""Question answering with checked citations (S3).

The generator (vLLM on MI300X, or the CPU stub) only proposes claims with quotes. This module decides what reaches the
user: a quote must appear verbatim (whitespace collapsed) in a document that retrieval returned for this question;
claims left without a valid quote are dropped; if retrieval is too weak, or nothing survives, the answer is a refusal.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .paths import REPO
from .retrieval import BM25, Doc

TAU_FILE = REPO / "artifacts" / "retrieval.json"
MIN_QUOTE = 20  # characters; stops "the" or "air" from counting as evidence
REFUSAL = "Not found in the manual. Please contact a senior engineer."
NO_VERIFIED = "No answer could be backed by a verbatim quote from the manual. Please contact a senior engineer."

Generate = Callable[[list[dict]], str]


def choose_tau(pos: list[float], neg: list[float]) -> tuple[float, int]:
    """Threshold on the top manual score: answer when score >= tau.

    Minimises (answerable questions refused + unanswerable questions answered); among equal-error gaps takes the widest,
    then the higher one, and returns its midpoint."""
    pts = sorted(set(pos) | set(neg))
    cands = [(pts[0] - 1.0, pts[0])] + list(zip(pts, pts[1:])) + [(pts[-1], pts[-1] + 1.0)]
    best = None
    for lo, hi in cands:
        t = (lo + hi) / 2
        err = sum(p < t for p in pos) + sum(n >= t for n in neg)
        key = (err, -(hi - lo), -t)
        if best is None or key < best[0]:
            best = (key, round(t, 3), err)
    return best[1], best[2]


def load_tau() -> float:
    """tau as chosen by scripts/eval_retrieval.py; missing file is an error, not a silent default."""
    return float(json.loads(TAU_FILE.read_text())["tau"])


def _norm(s: str) -> str:
    return " ".join(s.split())


def verify(claims: list[dict], docs: list[Doc]) -> list[dict]:
    """Keep only citations whose quote is verbatim in a retrieved doc; drop claims with none left."""
    by_key = {(d.source, d.id): _norm(d.text) for d in docs}
    kept = []
    for c in claims:
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        good = []
        for cit in c.get("citations") or []:
            if not isinstance(cit, dict):
                continue
            body = by_key.get((cit.get("source"), cit.get("id")))
            quote = cit.get("quote")
            if body is None or not isinstance(quote, str):
                continue
            q = _norm(quote)
            if len(q) >= MIN_QUOTE and q in body:
                good.append({"source": cit["source"], "id": cit["id"], "quote": q})
        if good:
            kept.append({"text": text.strip(), "citations": good})
    return kept


def build_messages(question: str, docs: list[Doc], context: str = "") -> list[dict]:
    blocks = "\n\n".join(f"[{d.source} {d.id}] {d.title}\n{d.text}" for d in docs)
    system = (
        "You are a maintenance assistant for the WD-200 air compressor. Answer only from the sources given. "
        "Reply with JSON only: {\"claims\": [{\"text\": \"one recommendation\", \"citations\": "
        "[{\"source\": \"manual\" or \"log\", \"id\": \"M-4.1 or L-001\", \"quote\": \"exact words copied from that source\"}]}]}. "
        "Every claim needs at least one citation, and every quote must be copied character for character. "
        "If the sources do not answer the question, reply {\"claims\": []}."
    )
    user = (f"Unit status: {context}\n\n" if context else "") + f"Sources:\n{blocks}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_claims(raw: str) -> list[dict]:
    """Model output to a list of claims; anything unparseable is an empty list (and so a refusal)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    claims = data.get("claims") if isinstance(data, dict) else None
    return [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []


def stub_generate(messages: list[dict]) -> str:
    """CPU stand-in for the model: quotes the first 'Action:' sentence (or first sentence) of each manual source."""
    user = messages[-1]["content"]
    claims = []
    for m in re.finditer(r"^\[manual (M-\d+\.\d+)\] [^\n]*\n(.*?)(?=^\[|\nQuestion:)", user, re.S | re.M):
        body = " ".join(m.group(2).split())
        sentences = re.split(r"(?<=\.)\s+", body)
        pick = next((s for s in sentences if s.startswith("Action:")), sentences[0])
        claims.append({"text": pick, "citations": [{"source": "manual", "id": m.group(1), "quote": pick}]})
    return json.dumps({"claims": claims[:2]})


REF = re.compile(r"\bM-6\.\d+\b")  # procedure sections (M-6): the steps a fault section tells you to follow


def follow_refs(docs: list[Doc], index: BM25, limit: int = 3) -> list[Doc]:
    """Procedure sections that the retrieved sections point to ("follow the tower check in M-6.2"), one hop, in order."""
    have = {d.id for d in docs}
    by_id = {d.id: d for d in index.docs if d.source == "manual"}
    out = []
    for d in docs:
        for ref in REF.findall(d.text):
            if ref not in have and ref in by_id and len(out) < limit:
                have.add(ref)
                out.append(by_id[ref])
    return out


def ask(question: str, index: BM25, generate: Generate, tau: float, context: str = "", k: int = 3,
        search: str | None = None, refs: bool = False) -> dict:
    """search: retrieval query when it should differ from the question; refs: also pass the sections they cite."""
    q = search or question
    manual = index.search(q, k=k, source="manual")
    top = manual[0][1] if manual else 0.0
    if top < tau:
        return {"answer": REFUSAL, "citations": [], "refused": True, "top_score": round(top, 3)}
    docs = [d for d, _ in manual]
    if refs:
        docs += follow_refs(docs, index)
    docs += [d for d, _ in index.search(q, k=2, source="log")]
    claims = verify(parse_claims(generate(build_messages(question, docs, context))), docs)
    if not claims:
        return {"answer": NO_VERIFIED, "citations": [], "refused": True, "top_score": round(top, 3)}
    answer = "\n".join(f"- {c['text']} [{', '.join(x['id'] for x in c['citations'])}]" for c in claims)
    cits = [x for c in claims for x in c["citations"]]
    return {"answer": answer, "citations": cits, "refused": False, "top_score": round(top, 3)}
