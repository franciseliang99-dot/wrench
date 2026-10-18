"""Retrieval hit rate on the 20 questions and the refusal threshold tau (S3); writes artifacts/retrieval.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrench.paths import REPO  # noqa: E402
from wrench.qa import choose_tau  # noqa: E402
from wrench.retrieval import BM25, load_corpus  # noqa: E402

QUESTIONS = REPO / "eval" / "retrieval_questions.json"
OUT = REPO / "artifacts" / "retrieval.json"


def main() -> int:
    spec = json.loads(QUESTIONS.read_text())
    index = BM25(load_corpus())
    rows = []
    for item in spec["in_scope"]:
        hits = index.search(item["q"], k=3, source="manual")
        ids = [d.id for d, _ in hits]
        rows.append({
            "q": item["q"], "expect": item["expect"], "top3": ids,
            "top_score": round(hits[0][1], 3) if hits else 0.0,
            "hit1": bool(ids[:1]) and ids[0] in item["expect"],
            "hit3": any(i in item["expect"] for i in ids),
        })
    neg = []
    for q in spec["out_of_scope"]:
        hits = index.search(q, k=1, source="manual")
        neg.append({"q": q, "top": hits[0][0].id if hits else None, "top_score": round(hits[0][1], 3) if hits else 0.0})
    tau, errors = choose_tau([r["top_score"] for r in rows], [n["top_score"] for n in neg])
    n = len(rows)
    report = {
        "questions": n,
        "hit_at_1": sum(r["hit1"] for r in rows),
        "hit_at_3": sum(r["hit3"] for r in rows),
        "tau": tau,
        "tau_errors": errors,
        "tau_rule": "midpoint of the widest gap that minimises (in-scope refused + out-of-scope answered); ties go to the higher tau",
        "in_scope": rows,
        "out_of_scope": neg,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"hit@1 {report['hit_at_1']}/{n} · hit@3 {report['hit_at_3']}/{n} · tau {tau} · errors {errors}")
    for r in rows:
        if not r["hit3"]:
            print(f"  miss: {r['q']!r} expect {r['expect']} got {r['top3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
