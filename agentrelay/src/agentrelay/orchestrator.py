"""Planner + DAG executor: validates the model's plan, runs independent steps in parallel,
isolates failures, and degrades gracefully instead of crashing the whole run."""
from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, AgentResult
from .llm import LLM, extract_json
from .tools import ApprovalDenied, Approver, ToolRegistry, deny_all
from .tracing import Tracer


class PlanError(ValueError):
    """The planner produced a plan that cannot be executed safely."""


@dataclass
class Step:
    id: str
    agent: str
    task: str
    params: dict[str, Any]
    depends_on: list[str]


@dataclass
class StepResult:
    id: str
    agent: str
    status: str  # completed | failed | denied | skipped
    text: str = ""
    tool: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class RunResult:
    goal: str
    status: str  # completed | partial | failed
    plan: list[Step]
    steps: dict[str, StepResult]
    final: str
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "status": self.status, "final": self.final, "trace_id": self.trace_id,
            "steps": {k: vars(v) for k, v in self.steps.items()},
        }


def validate_plan(raw: dict[str, Any], agents: dict[str, Agent], max_steps: int = 8) -> list[Step]:
    items = raw.get("steps")
    if not isinstance(items, list) or not items:
        raise PlanError("plan must contain a non-empty 'steps' list")
    if len(items) > max_steps:
        raise PlanError(f"plan has {len(items)} steps; limit is {max_steps}")
    steps: list[Step] = []
    for item in items:
        try:
            steps.append(Step(str(item["id"]), item["agent"], item.get("task", ""), item.get("params", {}),
                              list(item.get("depends_on", []))))
        except (KeyError, TypeError) as exc:
            raise PlanError(f"malformed step {item!r}: {exc}") from exc
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate step ids")
    for s in steps:
        if s.agent not in agents:
            raise PlanError(f"step {s.id} uses unknown agent '{s.agent}'")
        for dep in s.depends_on:
            if dep not in ids:
                raise PlanError(f"step {s.id} depends on unknown step '{dep}'")
    _layers(steps)  # raises PlanError on cycles
    return steps


def _layers(steps: list[Step]) -> list[list[Step]]:
    remaining, done, layers = {s.id: s for s in steps}, set(), []
    while remaining:
        ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
        if not ready:
            raise PlanError("plan contains a dependency cycle")
        layers.append(ready)
        done.update(s.id for s in ready)
        for s in ready:
            del remaining[s.id]
    return layers


class Orchestrator:
    def __init__(self, llm: LLM, agents: dict[str, Agent], registry: ToolRegistry, *,
                 approver: Approver = deny_all, tracer: Tracer | None = None,
                 max_steps: int = 8, max_workers: int = 4) -> None:
        self.llm, self.agents, self.registry = llm, agents, registry
        self.approver, self.tracer = approver, tracer or Tracer()
        self.max_steps, self.max_workers = max_steps, max_workers

    def plan(self, goal: str) -> list[Step]:
        catalog = [{"name": a.name, "description": a.description} for a in self.agents.values()]
        system = ("You are a planner. Break the goal into steps for the listed agents. Reply with JSON only: "
                  '{"steps":[{"id","agent","task","params","depends_on"}]}. Use {{step_id}} in params to pass results.')
        feedback = ""
        for attempt in (1, 2):  # one self-repair attempt when the plan is invalid
            with self.tracer.span("llm:plan", kind="llm", attempt=attempt):
                reply = self.llm.complete("plan", system, json.dumps({"goal": goal, "agents": catalog, "feedback": feedback}))
            try:
                return validate_plan(extract_json(reply), self.agents, self.max_steps)
            except (PlanError, ValueError) as exc:
                feedback = f"Previous plan was rejected: {exc}"
                if attempt == 2:
                    raise PlanError(str(exc)) from exc
        raise AssertionError("unreachable")

    def run(self, goal: str) -> RunResult:
        with self.tracer.span("run", kind="run", goal=goal):
            try:
                plan = self.plan(goal)
            except PlanError as exc:
                return RunResult(goal, "failed", [], {}, f"Could not build a valid plan: {exc}", self.tracer.trace_id)
            results: dict[str, StepResult] = {}
            for layer in _layers(plan):
                runnable = []
                for step in layer:
                    bad = [d for d in step.depends_on if results[d].status != "completed"]
                    if bad:
                        results[step.id] = StepResult(step.id, step.agent, "skipped", error=f"dependency not completed: {bad}")
                    else:
                        runnable.append(step)
                upstream = {k: r.text for k, r in results.items() if r.status == "completed"}
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = [
                        pool.submit(contextvars.copy_context().run, self._run_step, s, dict(upstream)) for s in runnable
                    ]
                    for fut in futures:
                        res = fut.result()
                        results[res.id] = res
            statuses = [r.status for r in results.values()]
            overall = "completed" if all(s == "completed" for s in statuses) else (
                "partial" if "completed" in statuses else "failed")
            final = self._write(goal, plan, results)
            return RunResult(goal, overall, plan, results, final, self.tracer.trace_id)

    def _run_step(self, step: Step, upstream: dict[str, str]) -> StepResult:
        agent = self.agents[step.agent]
        with self.tracer.span(f"agent:{agent.name}", kind="agent", step=step.id):
            try:
                res: AgentResult = agent.run(step.task, step.params, upstream, llm=self.llm, registry=self.registry,
                                             tracer=self.tracer, approver=self.approver)
                return StepResult(step.id, agent.name, "completed", res.text, res.tool, res.output)
            except ApprovalDenied as exc:
                return StepResult(step.id, agent.name, "denied", error=str(exc))
            except Exception as exc:  # isolate failures so sibling steps still finish
                return StepResult(step.id, agent.name, "failed", error=f"{type(exc).__name__}: {exc}")

    def _write(self, goal: str, plan: list[Step], results: dict[str, StepResult]) -> str:
        payload = {"goal": goal, "steps": [
            {"agent": results[s.id].agent, "status": results[s.id].status, "text": results[s.id].text,
             "error": results[s.id].error} for s in plan]}
        with self.tracer.span("llm:write", kind="llm"):
            return self.llm.complete("write", "Write a concise final report from the step results.", json.dumps(payload))
