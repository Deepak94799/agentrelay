"""Lightweight structured tracing: nested spans written to JSONL, with PII redaction."""
from __future__ import annotations

import contextvars
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_span", default=None)


def redact(value: Any) -> Any:
    """Mask e-mail addresses anywhere inside a JSON-like value before it is traced."""
    if isinstance(value, str):
        return _EMAIL.sub("[email]", value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


@dataclass
class Span:
    id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float
    duration_ms: float = 0.0
    status: str = "ok"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects spans in memory and optionally appends each finished span to a JSONL file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, name: str, kind: str = "step", **attrs: Any) -> Iterator[Span]:
        span = Span(
            id=uuid.uuid4().hex[:8],
            trace_id=self.trace_id,
            parent_id=_current_span.get(),
            name=name,
            kind=kind,
            start=time.time(),
            attrs=dict(attrs),
        )
        token = _current_span.set(span.id)
        began = time.perf_counter()
        try:
            yield span
        except BaseException as exc:  # record and re-raise
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - began) * 1000, 2)
            _current_span.reset(token)
            self._finish(span)

    def _finish(self, span: Span) -> None:
        span.attrs = redact(span.attrs)
        if span.error:
            span.error = redact(span.error)
        with self._lock:
            self.spans.append(span)
            if self._path:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(span), default=str) + "\n")

    def by_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]
