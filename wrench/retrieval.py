"""BM25 over manual sections and maintenance records: pure Python, no model, runs on the light host."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .paths import REPO

CORPUS = REPO / "corpus"
SECTION = re.compile(r"^## (M-\d+\.\d+) (.+)$", re.M)
STOP = frozenset(
    "a an the and or of to in on at for with is are be it this that what which how do does i my should "
    "when why can not no as by from if then than there its into about after before while".split()
)


@dataclass(frozen=True)
class Doc:
    source: str  # "manual" | "log"
    id: str
    title: str
    text: str


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return [_stem(w) for w in words if w not in STOP]


def load_manual(root: Path = CORPUS / "manual") -> list[Doc]:
    docs = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text()
        heads = list(SECTION.finditer(text))
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[m.end():end].strip()
            docs.append(Doc("manual", m.group(1), m.group(2), body))
    return docs


def load_logs(path: Path = CORPUS / "logs" / "maintenance.jsonl") -> list[Doc]:
    docs = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        text = f"Unit {r['unit_id']}, {r['date']}. Symptom: {r['symptom']} Action: {r['action']} Parts: {r['parts']}. Outcome: {r['outcome']}"
        docs.append(Doc("log", r["log_id"], f"{r['unit_id']} {r['date']}", text))
    return docs


def load_corpus() -> list[Doc]:
    return load_manual() + load_logs()


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d.title + " " + d.text)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

    def search(self, query: str, k: int = 3, source: str | None = None) -> list[tuple[Doc, float]]:
        ranked = sorted(
            ((d, s) for d, s in zip(self.docs, self.score(query)) if source is None or d.source == source),
            key=lambda x: (-x[1], x[0].id),
        )
        return [(d, s) for d, s in ranked[:k] if s > 0]
