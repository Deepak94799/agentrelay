"""Default tool set: knowledge-base search, sandboxed CSV analytics, approval-gated notifications."""
from __future__ import annotations

import csv
import json
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

from .rag import Retriever
from .tools import Tool, ToolError, ToolRegistry

_EMAIL = re.compile(r"^[\w.+-]+@([\w-]+\.[\w.-]+)$")


def build_registry(
    kb_dir: str | Path,
    data_dir: str | Path,
    outbox: str | Path,
    allowed_domains: tuple[str, ...] = ("example.com",),
    backoff_s: float = 0.2,
) -> ToolRegistry:
    retriever = Retriever.from_dir(kb_dir)
    data_root = Path(data_dir).resolve()
    outbox_path = Path(outbox)

    def search_kb(query: str, k: int = 3) -> dict:
        return {"hits": [{"source": h.source, "score": h.score, "text": h.text} for h in retriever.search(query, k)]}

    def query_csv(file: str, group_by: str, metric: str, agg: str = "sum") -> dict:
        path = (data_root / Path(file).name).resolve()  # basename only: blocks path traversal
        if data_root not in path.parents or path.suffix != ".csv" or not path.exists():
            raise ToolError(f"dataset not available: {file}")
        groups: dict[str, list[float]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for col in (group_by, metric):
                if col not in (reader.fieldnames or []):
                    raise ToolError(f"unknown column '{col}'; available: {reader.fieldnames}")
            for row in reader:
                groups[row[group_by]].append(float(row[metric]))
        reducers = {"sum": sum, "avg": lambda v: sum(v) / len(v), "count": len}
        rows = {k: round(reducers[agg](v), 2) for k, v in groups.items()}
        return {"group_by": group_by, "metric": metric, "agg": agg, "rows": dict(sorted(rows.items(), key=lambda kv: -kv[1]))}

    def send_notification(to: str, subject: str, body: str) -> dict:
        m = _EMAIL.match(to)
        if not m or m.group(1).lower() not in allowed_domains:
            raise ToolError(f"recipient domain not allowed: {to} (allowed: {', '.join(allowed_domains)})")
        record = {"id": uuid.uuid4().hex[:10], "ts": time.time(), "to": to, "subject": subject, "body": body}
        outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with outbox_path.open("a", encoding="utf-8") as fh:  # sandboxed: swap for SMTP/Slack/webhook in production
            fh.write(json.dumps(record) + "\n")
        return {"id": record["id"], "to": to}

    reg = ToolRegistry(backoff_s=backoff_s)
    reg.register(Tool(
        "search_kb", "Semantic-ish search over the internal knowledge base.",
        {"type": "object", "required": ["query"], "additionalProperties": False,
         "properties": {"query": {"type": "string", "maxLength": 500}, "k": {"type": "integer", "minimum": 1, "maximum": 10}}},
        search_kb))
    reg.register(Tool(
        "query_csv", "Group and aggregate a CSV dataset (sum, avg or count).",
        {"type": "object", "required": ["file", "group_by", "metric"], "additionalProperties": False,
         "properties": {"file": {"type": "string"}, "group_by": {"type": "string"}, "metric": {"type": "string"},
                        "agg": {"type": "string", "enum": ["sum", "avg", "count"]}}},
        query_csv))
    reg.register(Tool(
        "send_notification", "Send a message to a teammate. Side-effecting: requires human approval.",
        {"type": "object", "required": ["to", "subject", "body"], "additionalProperties": False,
         "properties": {"to": {"type": "string"}, "subject": {"type": "string", "maxLength": 200},
                        "body": {"type": "string", "maxLength": 4000}}},
        send_notification, requires_approval=True))
    return reg
