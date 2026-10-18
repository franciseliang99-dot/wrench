"""Record raw model outputs for the in-scope questions into artifacts/recorded.json (S6 replay).

Run with WRENCH_TF_KEY (AMD Token Factory) or the WRENCH_GPU_* variables set (see wrench/generation.py).
Outputs are stored raw; the API re-checks every citation when it replays them. Refuses to run with the CPU stub.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrench import qa  # noqa: E402
from wrench.generation import MI300X, RECORDED, TokenFactory, live_from_env, norm_question  # noqa: E402
from wrench.paths import REPO  # noqa: E402
from wrench.retrieval import BM25, load_corpus  # noqa: E402


PACE = 3.5


def main() -> int:
    live = live_from_env()
    if not isinstance(live, (TokenFactory, MI300X)):
        raise SystemExit("set WRENCH_TF_KEY or WRENCH_GPU_URL/KEY/CERT/MODEL (and not WRENCH_GENERATOR=stub): "
                         "recordings must come from a model on an AMD GPU")
    questions = [q["q"] for q in json.loads((REPO / "eval" / "retrieval_questions.json").read_text())["in_scope"]]
    index, tau = BM25(load_corpus()), qa.load_tau()
    answers, refused = {}, 0
    for q in questions:
        captured = {}

        def gen(messages):
            time.sleep(PACE)  # Token Factory allows 20 requests a minute per team key
            captured["output"] = live(messages)
            return captured["output"]

        r = qa.ask(q, index, gen, tau)
        refused += r["refused"]
        if "output" in captured:
            answers[norm_question(q)] = {"output": captured["output"],
                                         "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    machine = "AMD Token Factory (AMD GPUs)" if isinstance(live, TokenFactory) else "AMD Instinct MI300X"
    out = {"machine": machine, "model": live._model, "answers": answers}
    RECORDED.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(answers)} recorded, {refused} refused after the citation check -> {RECORDED.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
