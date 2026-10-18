"""HTTP API and static front end (§5.3, S4, S6, S8–S10).

Run:  uvicorn --factory wrench.api:app_from_env

Environment: see generation.py for the generator; WRENCH_DEMO_PASSCODE enables approval (unset = approval disabled);
WRENCH_ORIGIN is the one cross-site origin allowed (unset = none); WRENCH_TRUST_PROXY=1 takes the client address from the
rightmost X-Forwarded-For entry (set it only behind a proxy that appends that header).
"""
from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import qa
from .generation import GenerationOffline, Live, live_from_env, load_recorded, norm_question
from .paths import REPO
from .retrieval import BM25, load_corpus

ART = REPO / "artifacts"
TIMELINE = ART / "timeline.min.json"
LOGS = REPO / "corpus" / "logs" / "maintenance.jsonl"
MAX_QUESTION = 500
RATE_PER_MIN = 10
MAX_ORDERS = 200
AIR_SENSORS = {"TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "LPS", "Pressure_switch", "Towers", "DV_eletric", "COMP"}
LEVELS = {"o": "ok", "w": "watch", "a": "act", "-": "no data"}


class AskIn(BaseModel):
    unit_id: str
    ts: str
    question: str


class PointIn(BaseModel):
    unit_id: str
    ts: str


class Timeline:
    def __init__(self, data: dict):
        self.d = data
        self.start = datetime.fromisoformat(data["start"])
        self.step = timedelta(seconds=data["step_seconds"])

    def at(self, unit_id: str, ts: str) -> dict:
        if unit_id != self.d["unit_id"]:
            raise HTTPException(404, "unknown unit")
        try:
            t = datetime.fromisoformat(ts)
        except ValueError:
            raise HTTPException(422, "ts must be ISO 8601") from None
        off = t - self.start
        i = off // self.step
        if off % self.step or not 0 <= i < len(self.d["level"]):
            raise HTTPException(404, "ts is not a step of the timeline")
        drivers = [{"sensor": self.d["sensors"][s], "z": z, "direction": "high" if z > 0 else "low"}
                   for s, z in self.d["drivers"][i]]
        return {"unit_id": unit_id, "ts": t.isoformat(), "level": LEVELS[self.d["level"][i]],
                "anomaly_score": self.d["score"][i], "drivers": drivers}


def context_line(p: dict) -> str:
    drv = ", ".join(f"{d['sensor']} {d['direction']} (z={d['z']})" for d in p["drivers"]) or "none"
    return f"unit {p['unit_id']} at {p['ts']}: alert level {p['level']}; drivers: {drv}"


def priority(p: dict) -> tuple[int, str]:
    """M-7.2, deterministic. LPS among the drivers -> 1; act -> 2; watch -> 3 (M-3.3: file only if it repeats)."""
    sensors = {d["sensor"] for d in p["drivers"]}
    if "LPS" in sensors:
        return 1, "M-7.2: LPS event, inspect before the next run"
    if p["level"] == "act":
        air = "with an air-system driver" if sensors & AIR_SENSORS else "without an air-system driver (rule applied as act)"
        return 2, f"M-7.2: act alert {air}, inspect at the next depot stop"
    return 3, "M-7.2 / M-3.3: watch, file only if it repeats"


class RateLimiter:
    def __init__(self, per_minute: int, now: Callable[[], float]):
        self.n, self.now, self.hits = per_minute, now, defaultdict(deque)

    def allow(self, key: tuple) -> bool:
        t, q = self.now(), self.hits[key]
        while q and t - q[0] >= 60:
            q.popleft()
        if len(q) >= self.n:
            return False
        q.append(t)
        return True


def create_app(*, live: Live | None, recorded: dict, passcode: str | None = None, origin: str | None = None,
               trust_proxy: bool = False, rate: int = RATE_PER_MIN, now: Callable[[], float] = time.monotonic,
               timeline: dict | None = None, tau: float | None = None) -> FastAPI:
    app = FastAPI(title="Wrench", docs_url=None, redoc_url=None, openapi_url=None)
    if origin:
        app.add_middleware(CORSMiddleware, allow_origins=[origin], allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-Demo-Passcode"])
    tl = Timeline(timeline if timeline is not None else json.loads(TIMELINE.read_text()))
    index = BM25(load_corpus())
    tau = qa.load_tau() if tau is None else tau
    parts_by_log = {r["log_id"]: r["parts"] for r in map(json.loads, LOGS.read_text().splitlines())}
    answers = recorded.get("answers", {})
    limiter = RateLimiter(rate, now)
    orders: OrderedDict[str, dict] = OrderedDict()
    state = {"last_live_ok": None}

    def client_ip(req: Request) -> str:
        if trust_proxy:
            xff = req.headers.get("x-forwarded-for", "")
            last = xff.split(",")[-1].strip()
            if last:
                return last
        return req.client.host if req.client else "unknown"

    def guard(req: Request) -> None:
        if not limiter.allow((client_ip(req), req.url.path.split("/")[2])):
            raise HTTPException(429, "too many requests, try again in a minute")

    def run_live(messages: list[dict], served: dict) -> str:
        if live is None:
            raise GenerationOffline("no live generator configured")
        out = live(messages)
        served["served_by"] = live.name
        state["last_live_ok"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return out

    @app.exception_handler(GenerationOffline)
    async def offline(_req, _exc):
        return JSONResponse({"reason": "generation_offline"}, status_code=503)

    @app.get("/api/status")
    def status():
        return {"generator": live.name if live else "offline", "last_live_ok": state["last_live_ok"],
                "recorded_answers": len(answers), "recorded_machine": recorded.get("machine"), "tau": tau,
                "note": "Demo environment: work-order state resets when the server restarts."}

    @app.get("/api/timeline")
    def timeline_file():
        return FileResponse(TIMELINE, media_type="application/json")

    @app.get("/api/evaluation")
    def evaluation():
        r = json.loads((ART / "retrieval.json").read_text())
        return {"alerts": json.loads((ART / "evaluation.json").read_text()),
                "retrieval": {k: r[k] for k in ("questions", "hit_at_1", "hit_at_3", "tau", "tau_errors", "tau_rule")}}

    @app.get("/api/point")
    def point(unit_id: str, ts: str):
        return tl.at(unit_id, ts)

    @app.post("/api/ask")
    def ask(body: AskIn, req: Request):
        if len(body.question) > MAX_QUESTION:
            raise HTTPException(413, f"question longer than {MAX_QUESTION} characters")
        guard(req)
        p = tl.at(body.unit_id, body.ts)
        served: dict = {"served_by": "none"}

        def generate(messages):
            try:
                return run_live(messages, served)
            except GenerationOffline:
                rec = answers.get(norm_question(body.question))
                if not rec:
                    raise
                served.update(served_by="recorded", recorded_at=rec["recorded_at"],
                              recorded_on=recorded.get("machine", "unknown"))
                return rec["output"]

        r = qa.ask(body.question, index, generate, tau, context=context_line(p))
        return {**r, **served}

    @app.post("/api/workorder/draft")
    def draft(body: PointIn, req: Request):
        guard(req)
        p = tl.at(body.unit_id, body.ts)
        if p["level"] not in ("watch", "act"):
            raise HTTPException(409, f"no alert at this time (level {p['level']})")
        served: dict = {}
        q = f"{p['level']} alert. Drivers: " + ", ".join(f"{d['sensor']} {d['direction']}" for d in p["drivers"]) \
            + ". What should be inspected and done?"
        r = qa.ask(q, index, lambda m: run_live(m, served), tau, context=context_line(p))  # live only: never replayed
        prio, rule = priority(p)
        parts = sorted({parts_by_log[c["id"]] for c in r["citations"]
                        if c["source"] == "log" and parts_by_log.get(c["id"], "none") != "none"})
        order = {
            "id": uuid.uuid4().hex[:12], "unit_id": p["unit_id"], "alert_time": p["ts"], "level": p["level"],
            "drivers": p["drivers"], "priority": prio, "priority_rule": rule,
            "steps": r["answer"], "steps_verified": not r["refused"], "parts": parts, "safety": "M-6.3",
            "status": "draft", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        orders[order["id"]] = order
        while len(orders) > MAX_ORDERS:
            orders.popitem(last=False)
        return {"workorder": order, "citations": r["citations"], "served_by": served.get("served_by", "none")}

    @app.post("/api/workorder/{order_id}/approve")
    def approve(order_id: str, req: Request, x_demo_passcode: str | None = Header(default=None)):
        guard(req)
        if not passcode:
            raise HTTPException(403, "approval is disabled on this deployment")
        if not x_demo_passcode or not hmac.compare_digest(x_demo_passcode.encode(), passcode.encode()):
            raise HTTPException(403, "wrong demo passcode")
        order = orders.get(order_id)
        if order is None:
            raise HTTPException(404, "no such work order (state resets on restart)")
        order.update(status="approved", approved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return {"workorder": order}

    @app.get("/")
    def index_page():
        return FileResponse(REPO / "web" / "index.html")

    app.mount("/static", StaticFiles(directory=REPO / "web"), name="static")
    return app


def app_from_env() -> FastAPI:
    return create_app(live=live_from_env(), recorded=load_recorded(),
                      passcode=os.environ.get("WRENCH_DEMO_PASSCODE") or None,
                      origin=os.environ.get("WRENCH_ORIGIN") or None,
                      trust_proxy=os.environ.get("WRENCH_TRUST_PROXY") == "1")
