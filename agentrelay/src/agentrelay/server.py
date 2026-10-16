"""FastAPI service: health check, direct runs, and HMAC-signed webhooks that trigger runs."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .bootstrap import build_orchestrator
from .tools import auto_approve, deny_all


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=1000)


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def create_app(home: str | Path | None = None, webhook_secret: str | None = None, allow_side_effects: bool | None = None) -> FastAPI:
    secret = webhook_secret if webhook_secret is not None else os.environ.get("AGENTRELAY_WEBHOOK_SECRET", "")
    side_effects = allow_side_effects if allow_side_effects is not None else os.environ.get("AGENTRELAY_AUTO_APPROVE") == "1"
    app = FastAPI(title="AgentRelay", version="0.1.0")
    runs: dict[str, dict] = {}

    def execute(goal: str) -> dict:
        # a fresh orchestrator (and tracer) per run keeps traces isolated; side effects stay denied unless enabled
        orch = build_orchestrator(home, approver=auto_approve if side_effects else deny_all)
        result = orch.run(goal).to_dict()
        runs[result["trace_id"]] = result
        return result

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "side_effects_enabled": side_effects}

    @app.post("/runs")
    def create_run(req: RunRequest) -> dict:
        return execute(req.goal)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        if run_id not in runs:
            raise HTTPException(404, "unknown run")
        return runs[run_id]

    @app.post("/webhooks/{source}")
    async def webhook(source: str, request: Request, x_signature: str = Header(default="")) -> dict:
        if not secret:
            raise HTTPException(503, "webhook secret not configured")
        body = await request.body()
        if not hmac.compare_digest(sign(secret, body), x_signature):  # constant-time compare
            raise HTTPException(401, "invalid signature")
        try:
            goal = RunRequest(**json.loads(body)).goal
        except Exception as exc:
            raise HTTPException(422, f"body must be JSON with a 'goal': {exc}") from exc
        result = execute(goal)
        return {"source": source, "run_id": result["trace_id"], "status": result["status"]}

    return app

