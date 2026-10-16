from pathlib import Path

import pytest

from agentrelay.builtin_tools import build_registry
from agentrelay.rag import Retriever
from agentrelay.tools import ToolError, auto_approve
from agentrelay.tracing import Tracer

HOME = Path(__file__).parent.parent / "examples"


def test_retriever_ranks_the_right_document_first():
    r = Retriever.from_dir(HOME / "kb")
    assert r.search("refund annual plans")[0].source == "refund_policy.md"
    assert r.search("severity 1 incident postmortem")[0].source == "incident_runbook.md"
    assert r.search("zzzz qqqq") == []


def _reg(tmp_path):
    return build_registry(HOME / "kb", HOME / "data", tmp_path / "out.jsonl", backoff_s=0)


def _call(reg, name, args):
    return reg.call(name, args, agent="t", tracer=Tracer(), approver=auto_approve)


def test_csv_tool_aggregates_and_blocks_traversal(tmp_path):
    reg = _reg(tmp_path)
    out = _call(reg, "query_csv", {"file": "sales.csv", "group_by": "region", "metric": "revenue"})
    assert set(out["rows"]) == {"North", "South", "East", "West"}
    with pytest.raises(ToolError):
        _call(reg, "query_csv", {"file": "../../etc/passwd", "group_by": "a", "metric": "b"})
    with pytest.raises(ToolError, match="unknown column"):
        _call(reg, "query_csv", {"file": "sales.csv", "group_by": "nope", "metric": "revenue"})


def test_notification_domain_allowlist(tmp_path):
    reg = _reg(tmp_path)
    assert _call(reg, "send_notification", {"to": "a@example.com", "subject": "s", "body": "b"})["to"] == "a@example.com"
    assert (tmp_path / "out.jsonl").read_text().count("\n") == 1
    with pytest.raises(ToolError, match="domain not allowed"):
        _call(reg, "send_notification", {"to": "a@evil.com", "subject": "s", "body": "b"})
