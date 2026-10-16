import json
import time
from pathlib import Path

import pytest

from agentrelay import PlanError, auto_approve, build_orchestrator, default_agents, deny_all
from agentrelay.orchestrator import validate_plan
from agentrelay.tracing import Tracer

HOME = Path(__file__).parent.parent / "examples"


def make(approver=deny_all, **kw):
    return build_orchestrator(HOME, approver=approver, backoff_s=0, **kw)


def test_plan_validation_catches_cycles_unknown_agents_and_limits():
    agents = default_agents()
    step = lambda i, a="data", deps=(): {"id": i, "agent": a, "task": "t", "params": {}, "depends_on": list(deps)}  # noqa: E731
    with pytest.raises(PlanError, match="cycle"):
        validate_plan({"steps": [step("a", deps=["b"]), step("b", deps=["a"])]}, agents)
    with pytest.raises(PlanError, match="unknown agent"):
        validate_plan({"steps": [step("a", "root")]}, agents)
    with pytest.raises(PlanError, match="unknown step"):
        validate_plan({"steps": [step("a", deps=["zz"])]}, agents)
    with pytest.raises(PlanError, match="limit"):
        validate_plan({"steps": [step(str(i)) for i in range(9)]}, agents)


def test_planner_repairs_an_invalid_first_plan():
    good = json.dumps({"steps": [{"id": "s1", "agent": "research", "task": "t", "params": {"query": "refund"}, "depends_on": []}]})
    replies = iter([json.dumps({"steps": [{"id": "s1", "agent": "hacker", "task": "", "params": {}}]}), good])
    seen = []

    class FlakyPlanner:
        def complete(self, role, system, prompt):
            if role == "plan":
                seen.append(json.loads(prompt)["feedback"])
                return next(replies)
            from agentrelay.llm import RuleBasedLLM
            return RuleBasedLLM().complete(role, system, prompt)

    run = make(llm=FlakyPlanner()).run("refund?")
    assert run.status == "completed"
    assert seen[0] == "" and "unknown agent" in seen[1]  # the second attempt saw why the first was rejected


def test_unrecoverable_plan_fails_cleanly():
    class Broken:
        def complete(self, role, system, prompt):
            return "not json at all"

    run = make(llm=Broken()).run("anything")
    assert run.status == "failed" and "valid plan" in run.final


def test_end_to_end_with_approval_and_handoff(tmp_path):
    run = make(auto_approve).run("Summarize sales by region and email the result to lead@example.com")
    assert run.status == "completed"
    assert {s.agent for s in run.plan} == {"data", "action"}
    outbox = HOME / "outbox.jsonl"
    try:
        sent = json.loads(outbox.read_text().splitlines()[-1])
        assert "revenue by region" in sent["body"]  # data step's output was handed to the action step
    finally:
        outbox.unlink(missing_ok=True)


def test_denied_action_degrades_to_partial_and_sends_nothing():
    outbox = HOME / "outbox.jsonl"
    outbox.unlink(missing_ok=True)
    run = make(deny_all).run("Summarize sales by region and email the result to lead@example.com")
    assert run.status == "partial"
    assert run.steps["s2"].status == "denied" and run.steps["s1"].status == "completed"
    assert not outbox.exists()


def test_failure_skips_dependents_but_not_siblings():
    run = make(auto_approve).run("Summarize sales by region and email the result to someone@evil.com")
    assert run.status == "partial"
    assert run.steps["s2"].status == "failed" and "domain not allowed" in run.steps["s2"].error


def test_independent_steps_run_in_parallel():
    orch = make()
    orig = orch.registry.get("search_kb").fn
    orch.registry.get("search_kb").fn = lambda **kw: (time.sleep(0.3), orig(**kw))[1]
    orig_csv = orch.registry.get("query_csv").fn
    orch.registry.get("query_csv").fn = lambda **kw: (time.sleep(0.3), orig_csv(**kw))[1]
    began = time.perf_counter()
    run = orch.run("Compare revenue by region and remind me of the pricing")
    assert run.status == "completed"
    assert time.perf_counter() - began < 0.55  # two 0.3s steps overlapped instead of running back to back


def test_trace_has_nested_spans_and_redacts_emails(tmp_path):
    trace = tmp_path / "t.jsonl"
    orch = make(auto_approve, trace_path=trace)
    orch.run("Summarize sales by region and email the result to lead@example.com")
    outbox = HOME / "outbox.jsonl"
    outbox.unlink(missing_ok=True)
    lines = [json.loads(l) for l in trace.read_text().splitlines()]
    kinds = {l["kind"] for l in lines}
    assert {"run", "agent", "tool", "llm"} <= kinds
    assert "lead@example.com" not in trace.read_text()
    run_span = next(l for l in lines if l["kind"] == "run")
    assert all(l["trace_id"] == run_span["trace_id"] for l in lines)
    assert any(l["parent_id"] == run_span["id"] for l in lines)
