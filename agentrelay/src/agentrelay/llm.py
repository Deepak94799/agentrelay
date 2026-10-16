"""LLM interface plus two implementations.

* ``RuleBasedLLM``  - deterministic and offline. Makes tests, CI and evals reproducible with no API key.
* ``AnthropicLLM``  - optional adapter for a real model (``pip install agentrelay[anthropic]``).

Every call carries a ``role`` (plan / act / summarize / write) so any provider can be swapped in.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLM(Protocol):
    def complete(self, role: str, system: str, prompt: str) -> str: ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates prose or code fences around it)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the extra: pip install 'agentrelay[anthropic]'") from exc
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model or os.environ.get("AGENTRELAY_MODEL", "claude-sonnet-5-5")
        self._max_tokens = max_tokens

    def complete(self, role: str, system: str, prompt: str) -> str:  # pragma: no cover - needs network
        msg = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


class RuleBasedLLM:
    """Deterministic stand-in that follows the same JSON contracts a real model must follow."""

    _DATA = ("sales", "revenue", "units", "region", "product")
    _KB = ("policy", "refund", "runbook", "incident", "pricing", "price", "escalat", "sla", "outage", "how do", "what is")
    _ACT = ("email", "send", "notify", "mail")

    def complete(self, role: str, system: str, prompt: str) -> str:
        payload = json.loads(prompt)
        return getattr(self, f"_{role}")(payload)

    def _plan(self, p: dict) -> str:
        goal, g = p["goal"], p["goal"].lower()
        wants_data = any(w in g for w in self._DATA)
        wants_act = any(w in g for w in self._ACT)
        wants_kb = any(w in g for w in self._KB) or not (wants_data or wants_act)
        steps: list[dict] = []
        if wants_data:
            steps.append({
                "id": "s1", "agent": "data", "task": "Aggregate the sales dataset", "depends_on": [],
                "params": {"file": "sales.csv", "group_by": "product" if "product" in g else "region",
                           "metric": "units" if "units" in g else "revenue",
                           "agg": "avg" if ("average" in g or "avg" in g) else "sum"}})
        if wants_kb:
            steps.append({"id": f"s{len(steps) + 1}", "agent": "research", "task": "Find relevant internal knowledge",
                          "depends_on": [], "params": {"query": goal}})
        if wants_act:
            ids = [s["id"] for s in steps]
            m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", goal)
            steps.append({"id": f"s{len(steps) + 1}", "agent": "action", "task": "Notify the requester", "depends_on": ids,
                          "params": {"to": m.group(0) if m else "team@example.com", "subject": "AgentRelay update",
                                     "body": "\n".join("{{%s}}" % i for i in ids) or goal}})
        return json.dumps({"steps": steps})

    def _act(self, p: dict) -> str:
        args = dict(p["params"])
        if p["default_tool"] == "search_kb":
            args.setdefault("query", p["task"])
        return json.dumps({"tool": p["default_tool"], "args": args})

    def _summarize(self, p: dict) -> str:
        r = p["result"]
        if p["tool"] == "search_kb":
            if not r["hits"]:
                return "No relevant knowledge-base entries found."
            best = r["hits"][0]["score"]
            keep = [h for h in r["hits"] if h["score"] >= 0.6 * best]  # passages close to the top score
            body = " ".join(re.sub(r"^#.*\n", "", h["text"]).replace("\n", " ") for h in keep)
            return f"{body} (sources: {', '.join(sorted({h['source'] for h in keep}))})"
        if p["tool"] == "query_csv":
            body = ", ".join(f"{k}={v:,}" for k, v in r["rows"].items())
            return f"{r['metric']} by {r['group_by']} ({r['agg']}): {body}"
        if p["tool"] == "send_notification":
            return f"Notification {r['id']} sent to {r['to']}."
        return json.dumps(r)

    def _write(self, p: dict) -> str:
        lines = [f"Goal: {p['goal']}", ""]
        lines += [f"[{s['agent']}] {s['text']}" for s in p["steps"] if s["status"] == "completed"]
        problems = [f"[{s['agent']}] {s['status']}: {s.get('error', '')}" for s in p["steps"] if s["status"] != "completed"]
        if problems:
            lines += ["", "Issues:", *problems]
        return "\n".join(lines)
