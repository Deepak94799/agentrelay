"""Command line: ``agentrelay run|eval|serve``."""
from __future__ import annotations

import argparse
import sys

from .bootstrap import build_orchestrator
from .evals import format_report, run_eval
from .tools import ask_on_console, auto_approve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentrelay", description="Multi-agent orchestration with guardrails and evals.")
    p.add_argument("--home", default=None, help="directory with kb/, data/ (default: ./examples or $AGENTRELAY_HOME)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="plan and execute a goal")
    r.add_argument("goal")
    r.add_argument("--yes", action="store_true", help="auto-approve side-effecting tools")
    r.add_argument("--trace", default=None, help="write spans to this JSONL file")

    e = sub.add_parser("eval", help="score the system on labelled cases")
    e.add_argument("--cases", default="evals/cases.jsonl")
    e.add_argument("--min-success", type=float, default=1.0, help="exit non-zero below this task success rate")

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    home = a.home or "examples"

    if a.cmd == "run":
        orch = build_orchestrator(home, approver=auto_approve if a.yes else ask_on_console, trace_path=a.trace)
        result = orch.run(a.goal)
        print(result.final)
        print(f"\nstatus={result.status}  steps={len(result.steps)}  trace_id={result.trace_id}")
        return 0 if result.status == "completed" else 1

    if a.cmd == "eval":
        report = run_eval(a.cases, home)
        print(format_report(report))
        return 0 if report["task_success"] >= a.min_success else 1

    import uvicorn  # imported lazily: the server is an optional extra
    from .server import create_app
    uvicorn.run(create_app(home), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
