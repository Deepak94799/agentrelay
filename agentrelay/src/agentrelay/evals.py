"""Evaluation harness: replays labelled goals and scores planning, routing, tool use and outcomes."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .bootstrap import build_orchestrator
from .tools import auto_approve, deny_all


@dataclass
class CaseResult:
    id: str
    passed: bool
    routing_ok: bool
    tools_ok: bool
    status_ok: bool
    content_ok: bool
    latency_ms: float
    detail: str = ""


def run_case(case: dict, home: str | Path) -> CaseResult:
    # run on a throw-away copy of the example home so evals never touch real output files
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "home"
        shutil.copytree(home, sandbox, ignore=shutil.ignore_patterns("outbox.jsonl"))
        orch = build_orchestrator(sandbox, approver=auto_approve if case.get("auto_approve") else deny_all,
                                  trace_path=Path(tmp) / "trace.jsonl", backoff_s=0.0)
        began = time.perf_counter()
        run = orch.run(case["goal"])
        latency = round((time.perf_counter() - began) * 1000, 1)
    agents_used = sorted({s.agent for s in run.plan})
    tools_used = sorted({r.tool for r in run.steps.values() if r.status == "completed" and r.tool})
    routing = agents_used == sorted(case.get("expect_agents", agents_used))
    tools = tools_used == sorted(case.get("expect_tools", tools_used))
    status = run.status == case.get("expect_status", "completed")
    content = all(s.lower() in run.final.lower() for s in case.get("expect_contains", []))
    detail = "" if (routing and tools and status and content) else (
        f"agents={agents_used} tools={tools_used} status={run.status}")
    return CaseResult(case["id"], routing and tools and status and content, routing, tools, status, content, latency, detail)


def run_eval(cases_path: str | Path, home: str | Path) -> dict:
    cases = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    results = [run_case(c, home) for c in cases]
    n = len(results)
    return {
        "cases": n,
        "task_success": sum(r.passed for r in results) / n,
        "routing_accuracy": sum(r.routing_ok for r in results) / n,
        "tool_accuracy": sum(r.tools_ok for r in results) / n,
        "status_accuracy": sum(r.status_ok for r in results) / n,
        "p50_latency_ms": sorted(r.latency_ms for r in results)[n // 2],
        "results": [vars(r) for r in results],
    }


def format_report(report: dict) -> str:
    lines = [f"{'case':<28}{'pass':<6}{'route':<7}{'tools':<7}{'status':<8}{'ms':>8}"]
    for r in report["results"]:
        mark = lambda b: "ok" if b else "FAIL"  # noqa: E731
        lines.append(f"{r['id']:<28}{mark(r['passed']):<6}{mark(r['routing_ok']):<7}{mark(r['tools_ok']):<7}"
                     f"{mark(r['status_ok']):<8}{r['latency_ms']:>8}" + (f"  <- {r['detail']}" if r["detail"] else ""))
    lines += ["", f"cases={report['cases']}  task_success={report['task_success']:.0%}  "
              f"routing={report['routing_accuracy']:.0%}  tools={report['tool_accuracy']:.0%}  "
              f"p50={report['p50_latency_ms']}ms"]
    return "\n".join(lines)
