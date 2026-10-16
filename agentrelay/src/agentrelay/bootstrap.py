"""Wire the default agents, tools and LLM into a ready-to-run Orchestrator."""
from __future__ import annotations

import os
from pathlib import Path

from .agents import default_agents
from .builtin_tools import build_registry
from .llm import LLM, AnthropicLLM, RuleBasedLLM
from .orchestrator import Orchestrator
from .tools import Approver, deny_all
from .tracing import Tracer


def build_orchestrator(
    home: str | Path | None = None,
    *,
    approver: Approver = deny_all,
    trace_path: str | Path | None = None,
    llm: LLM | None = None,
    backoff_s: float = 0.2,
) -> Orchestrator:
    home = Path(home or os.environ.get("AGENTRELAY_HOME", "examples"))
    if llm is None:
        llm = AnthropicLLM() if os.environ.get("AGENTRELAY_LLM", "offline") == "anthropic" else RuleBasedLLM()
    registry = build_registry(home / "kb", home / "data", home / "outbox.jsonl", backoff_s=backoff_s)
    return Orchestrator(llm, default_agents(), registry, approver=approver, tracer=Tracer(trace_path))
