import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentrelay.evals import run_eval
from agentrelay.server import create_app, sign

ROOT = Path(__file__).parent.parent
SECRET = "s3cret"


def client():
    return TestClient(create_app(ROOT / "examples", webhook_secret=SECRET, allow_side_effects=False))


def test_health_and_direct_run():
    c = client()
    assert c.get("/healthz").json()["side_effects_enabled"] is False
    body = c.post("/runs", json={"goal": "What is our refund policy?"}).json()
    assert body["status"] == "completed" and "30 days" in body["final"]
    assert c.get(f"/runs/{body['trace_id']}").status_code == 200
    assert c.get("/runs/nope").status_code == 404


def test_webhook_requires_valid_signature():
    c = client()
    raw = json.dumps({"goal": "What is our refund policy?"}).encode()
    assert c.post("/webhooks/github", content=raw).status_code == 401
    assert c.post("/webhooks/github", content=raw, headers={"X-Signature": "sha256=bad"}).status_code == 401
    ok = c.post("/webhooks/github", content=raw, headers={"X-Signature": sign(SECRET, raw)})
    assert ok.status_code == 200 and ok.json()["status"] == "completed"


def test_webhook_rejects_bad_payload_and_missing_secret():
    c = client()
    raw = b'{"nope": 1}'
    assert c.post("/webhooks/x", content=raw, headers={"X-Signature": sign(SECRET, raw)}).status_code == 422
    no_secret = TestClient(create_app(ROOT / "examples", webhook_secret=""))
    assert no_secret.post("/webhooks/x", content=raw).status_code == 503


def test_server_denies_side_effects_by_default():
    body = client().post("/runs", json={"goal": "Summarize sales by region and email it to a@example.com"}).json()
    assert body["status"] == "partial"


def test_eval_suite_meets_the_quality_bar():
    report = run_eval(ROOT / "evals" / "cases.jsonl", ROOT / "examples")
    failed = [r for r in report["results"] if not r["passed"]]
    assert not failed, failed
    assert report["task_success"] == 1.0
