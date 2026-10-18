import json
import re

import pytest
from fastapi.testclient import TestClient

from wrench import api, generation, qa
from wrench.generation import GenerationOffline, Live
from wrench.paths import REPO

TL = json.loads(api.TIMELINE.read_text())
Q = "Reservoir pressure keeps falling when the train is parked overnight, what should I check?"


def ts_with(level_letter):
    t = api.Timeline(TL)
    return (t.start + TL["level"].index(level_letter) * t.step).isoformat()


class Offline(Live):
    def __init__(self):
        super().__init__("mi300x")
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        raise GenerationOffline("ConnectError")


class Boom(Live):
    def __init__(self):
        super().__init__("mi300x")

    def __call__(self, messages):
        raise AssertionError("generator must not be called")


def client(live=None, recorded=None, **kw):
    return TestClient(api.create_app(live=live, recorded=recorded or {}, **kw))


def ask(c, question=Q, ts=None, **headers):
    return c.post("/api/ask", json={"unit_id": TL["unit_id"], "ts": ts or ts_with("o"), "question": question}, headers=headers)


def test_stub_answer_has_verbatim_citations():
    r = ask(client(generation.Stub()))
    d = r.json()
    assert r.status_code == 200 and d["served_by"] == "cpu-stub" and not d["refused"] and d["citations"]


def test_below_tau_refuses_without_generating():
    r = ask(client(Boom()), "How do I book annual leave?")
    assert r.status_code == 200 and r.json()["refused"] and r.json()["served_by"] == "none"


def test_offline_without_recording_is_503():
    r = ask(client(Offline()))
    assert r.status_code == 503 and r.json() == {"reason": "generation_offline"}


def test_offline_with_recording_replays_and_says_so():
    top = [d for d, _ in api.BM25(api.load_corpus()).search(Q, k=3, source="manual")]
    out = qa.stub_generate(qa.build_messages(Q, top))
    rec = {"machine": "MI300X", "answers": {generation.norm_question(Q): {"output": out, "recorded_at": "2026-10-20T10:00:00Z"}}}
    d = ask(client(Offline(), rec)).json()
    assert d["served_by"] == "recorded" and d["recorded_at"] == "2026-10-20T10:00:00Z" and d["recorded_on"] == "MI300X"
    assert not d["refused"] and d["citations"]


def test_replayed_output_is_still_checked():
    fake = json.dumps({"claims": [{"text": "x", "citations": [{"source": "manual", "id": "M-4.1", "quote": "Raise the setpoints to hide the leak entirely."}]}]})
    rec = {"answers": {generation.norm_question(Q): {"output": fake, "recorded_at": "t"}}}
    d = ask(client(Offline(), rec)).json()
    assert d["refused"] and d["answer"] == qa.NO_VERIFIED


def test_workorder_is_never_replayed():
    class Everything(dict):  # a recording for every possible question
        def get(self, k, default=None):
            return {"output": qa.stub_generate(qa.build_messages(k, [])), "recorded_at": "t"}
    rec = {"answers": Everything(x=1)}
    live = Offline()
    r = client(live, rec).post("/api/workorder/draft", json={"unit_id": TL["unit_id"], "ts": ts_with("a")})
    assert r.status_code == 503 and live.calls == 1


def test_workorder_needs_an_alert():
    r = client(generation.Stub()).post("/api/workorder/draft", json={"unit_id": TL["unit_id"], "ts": ts_with("o")})
    assert r.status_code == 409


def test_workorder_draft_then_approve_only_changes_status():
    c = client(generation.Stub(), passcode="letmein")
    r = c.post("/api/workorder/draft", json={"unit_id": TL["unit_id"], "ts": ts_with("a")})
    o = r.json()["workorder"]
    assert r.status_code == 200 and o["status"] == "draft" and o["priority"] in (1, 2) and o["safety"] == "M-6.3"
    assert c.post(f"/api/workorder/{o['id']}/approve").status_code == 403
    assert c.post(f"/api/workorder/{o['id']}/approve", headers={"X-Demo-Passcode": "wrong"}).status_code == 403
    a = c.post(f"/api/workorder/{o['id']}/approve", headers={"X-Demo-Passcode": "letmein"}).json()["workorder"]
    assert a["status"] == "approved" and {k: v for k, v in a.items() if k not in ("status", "approved_at")} == \
        {k: v for k, v in o.items() if k != "status"}


def test_approve_is_disabled_without_passcode():
    c = client(generation.Stub())
    o = c.post("/api/workorder/draft", json={"unit_id": TL["unit_id"], "ts": ts_with("a")}).json()["workorder"]
    assert c.post(f"/api/workorder/{o['id']}/approve", headers={"X-Demo-Passcode": ""}).status_code == 403


@pytest.mark.parametrize("level,drivers,want", [
    ("act", ["LPS", "TP2"], 1), ("watch", ["LPS"], 1), ("act", ["TP2"], 2), ("act", ["Oil_temperature"], 2), ("watch", ["TP2"], 3),
])
def test_priority_follows_m72(level, drivers, want):
    assert api.priority({"level": level, "drivers": [{"sensor": s} for s in drivers]})[0] == want


def test_question_length_limit():
    assert ask(client(Boom()), "x" * 501).status_code == 413


def test_rate_limit_per_ip_and_rightmost_forwarded_for():
    t = [0.0]
    c = client(generation.Stub(), trust_proxy=True, rate=2, now=lambda: t[0])
    h = {"X-Forwarded-For": "6.6.6.6, 1.1.1.1"}
    assert [ask(c, **h).status_code for _ in range(3)] == [200, 200, 429]
    assert ask(c, **{"X-Forwarded-For": "1.1.1.1, 2.2.2.2"}).status_code == 200  # leftmost is client-controlled
    assert ask(c, **{"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 429
    t[0] = 61.0
    assert ask(c, **h).status_code == 200


def test_forwarded_for_ignored_without_trust_proxy():
    c = client(generation.Stub(), rate=1)
    assert ask(c, **{"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert ask(c, **{"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_cors_only_for_the_configured_origin():
    c = client(origin="https://wrench.example")
    good = c.get("/api/status", headers={"Origin": "https://wrench.example"})
    bad = c.get("/api/status", headers={"Origin": "https://evil.example"})
    assert good.headers.get("access-control-allow-origin") == "https://wrench.example"
    assert "access-control-allow-origin" not in bad.headers
    assert "access-control-allow-origin" not in client().get("/api/status", headers={"Origin": "https://wrench.example"}).headers


def test_point_rejects_off_grid_and_unknown_unit():
    c = client()
    assert c.get("/api/point", params={"unit_id": TL["unit_id"], "ts": "2020-02-01T00:05:00"}).status_code == 404
    assert c.get("/api/point", params={"unit_id": "nope", "ts": TL["start"]}).status_code == 404


def test_mi300x_errors_become_offline_without_the_key(tmp_path):
    import ssl
    import subprocess
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "1",
                    "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"], check=True, capture_output=True)
    ssl.create_default_context(cafile=str(cert))
    g = generation.MI300X("https://127.0.0.1:9", "sekrit-key-123", str(cert), "m")
    with pytest.raises(GenerationOffline) as e:
        g([{"role": "user", "content": "hi"}])
    assert "sekrit" not in str(e.value)


def test_live_from_env_requires_all_gpu_settings():
    assert generation.live_from_env({}) is None
    assert generation.live_from_env({"WRENCH_GENERATOR": "stub"}).name == "cpu-stub"
    with pytest.raises(ValueError):
        generation.live_from_env({"WRENCH_GPU_URL": "https://1.2.3.4"})
    with pytest.raises(ValueError):
        generation.MI300X("http://1.2.3.4", "k", "c", "m")


def test_front_end_never_parses_strings_as_html():
    js = (REPO / "web" / "app.js").read_text()
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(|new Function", js)


def test_no_endpoint_sends_anything_out():
    import ast
    tree = ast.parse((REPO / "wrench" / "api.py").read_text())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods & {"httpx", "requests", "urllib", "smtplib", "socket", "http"} == set()


def test_tokenfactory_posts_openai_body_and_hides_the_key():
    import httpx
    seen = {}

    def handler(req):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    g = generation.TokenFactory("rc-sekrit", "Gemma-4-31B-IT", transport=httpx.MockTransport(handler))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert g(msgs) == "ok" and g.name == "amd-tokenfactory"
    assert seen["url"] == "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
    assert seen["auth"] == "Bearer rc-sekrit"
    assert seen["body"] == {"model": "Gemma-4-31B-IT", "messages": msgs, "max_tokens": generation.MAX_TOKENS, "temperature": 0}


@pytest.mark.parametrize("resp", [
    lambda httpx: httpx.Response(429, json={"choices": [{"message": {"content": "rate limited rc-sekrit"}}]}),
    lambda httpx: httpx.Response(500, text="boom"),
    lambda httpx: httpx.Response(200, json={"choices": []}),
    lambda httpx: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
    lambda httpx: httpx.Response(200, text="not json"),
])
def test_tokenfactory_failures_become_offline_without_the_key(resp):
    import httpx
    g = generation.TokenFactory("rc-sekrit", transport=httpx.MockTransport(lambda req: resp(httpx)))
    with pytest.raises(GenerationOffline) as e:
        g([{"role": "user", "content": "hi"}])
    assert "sekrit" not in str(e.value)


def test_tokenfactory_failure_falls_back_to_the_recording():
    import httpx
    g = generation.TokenFactory("k", transport=httpx.MockTransport(lambda req: httpx.Response(429)))
    top = [d for d, _ in api.BM25(api.load_corpus()).search(Q, k=3, source="manual")]
    out = qa.stub_generate(qa.build_messages(Q, top))
    rec = {"machine": "AMD Token Factory (AMD GPUs)", "answers": {generation.norm_question(Q): {"output": out, "recorded_at": "t"}}}
    d = ask(client(g, rec)).json()
    assert d["served_by"] == "recorded" and d["recorded_on"] == "AMD Token Factory (AMD GPUs)"


def test_live_from_env_prefers_tokenfactory():
    assert generation.live_from_env({"WRENCH_TF_KEY": "k"}).name == "amd-tokenfactory"
    assert generation.live_from_env({"WRENCH_TF_KEY": "k", "WRENCH_TF_MODEL": "DeepSeek-V4-Flash"})._model == "DeepSeek-V4-Flash"
    assert generation.live_from_env({"WRENCH_TF_KEY": "k", "WRENCH_GENERATOR": "stub"}).name == "cpu-stub"
