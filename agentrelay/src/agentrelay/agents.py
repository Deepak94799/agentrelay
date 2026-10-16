"""Specialist agents. Each one has a narrow job and a least-privilege tool allowlist."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM, extract_json
from .tools import Approver, ToolError, ToolRegistry
from .tracing import Tracer

_TEMPLATE = re.compile(r"\{\{(\w+)\}\}")


@dataclass
class AgentResult:
    agent: str
    tool: str
    output: dict[str, Any]
    text: str


@dataclass
class Agent:
    name: str
    description: str
    allowed_tools: list[str]
    default_tool: str
    system: str = field(default="")

    def run(self, task: str, params: dict[str, Any], upstream: dict[str, str], *, llm: LLM,
            registry: ToolRegistry, tracer: Tracer, approver: Approver) -> AgentResult:
        params = _render(params, upstream)  # hand-off: inject earlier steps' text into this step's inputs
        system = self.system or f"You are the {self.name} agent. {self.description} Reply with JSON only."
        prompt = json.dumps({
            "agent": self.name, "task": task, "params": params, "default_tool": self.default_tool,
            "tools": registry.specs(self.allowed_tools),
        })
        with tracer.span("llm:act", kind="llm", agent=self.name):
            decision = extract_json(llm.complete("act", system, prompt))
        tool = decision.get("tool")
        if tool not in self.allowed_tools:  # least privilege: a model cannot talk an agent into other tools
            raise ToolError(f"agent '{self.name}' is not allowed to use tool '{tool}'")
        output = registry.call(tool, decision.get("args", {}), agent=self.name, tracer=tracer, approver=approver)
        with tracer.span("llm:summarize", kind="llm", agent=self.name):
            text = llm.complete("summarize", system, json.dumps({"task": task, "tool": tool, "result": output}))
        return AgentResult(self.name, tool, output, text)


def _render(value: Any, upstream: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _TEMPLATE.sub(lambda m: upstream.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _render(v, upstream) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, upstream) for v in value]
    return value


def default_agents() -> dict[str, Agent]:
    agents = [
        Agent("research", "Answers questions from the internal knowledge base.", ["search_kb"], "search_kb"),
        Agent("data", "Aggregates and analyses CSV datasets.", ["query_csv"], "query_csv"),
        Agent("action", "Performs side-effecting actions such as notifications; every action needs approval.",
              ["send_notification"], "send_notification"),
    ]
    return {a.name: a for a in agents}
