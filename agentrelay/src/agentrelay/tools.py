"""Tool registry with JSON-schema validation, approval gating, timeouts and retries."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from .tracing import Tracer


class ToolError(Exception):
    """Permanent tool failure (not retried)."""


class TransientToolError(ToolError):
    """Temporary failure; the registry retries these with exponential backoff."""


class ToolValidationError(ToolError):
    """Arguments did not match the tool's JSON schema."""


class ApprovalDenied(ToolError):
    """A human (or policy) refused a side-effecting tool call."""


@dataclass(frozen=True)
class ApprovalRequest:
    agent: str
    tool: str
    args: dict[str, Any]


Approver = Callable[[ApprovalRequest], bool]


def auto_approve(_: ApprovalRequest) -> bool:
    return True


def deny_all(_: ApprovalRequest) -> bool:
    return False


def ask_on_console(req: ApprovalRequest) -> bool:
    print(f"\n[approval needed] agent={req.agent} tool={req.tool}\n  args={req.args}")
    return input("  approve? [y/N] ").strip().lower() in {"y", "yes"}


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def validate(value: Any, schema: dict[str, Any], path: str = "args") -> None:
    """Validate against the JSON-Schema subset we need: type, enum, required, properties, bounds."""
    expected = schema.get("type")
    if expected:
        py = _TYPES[expected]
        bad_bool = isinstance(value, bool) and expected in {"integer", "number"}
        if not isinstance(value, py) or bad_bool:
            raise ToolValidationError(f"{path}: expected {expected}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        raise ToolValidationError(f"{path}: longer than {schema['maxLength']} characters")
    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolValidationError(f"{path}: missing required field '{key}'")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise ToolValidationError(f"{path}: unexpected fields {sorted(extra)}")
        for key, sub in props.items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., Any]
    requires_approval: bool = False
    timeout_s: float = 10.0
    retries: int = 2


class ToolRegistry:
    def __init__(self, backoff_s: float = 0.2) -> None:
        self._tools: dict[str, Tool] = {}
        self.backoff_s = backoff_s

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "schema": t.schema, "requires_approval": t.requires_approval}
            for t in (self.get(n) for n in names)
        ]

    def call(self, name: str, args: dict[str, Any], *, agent: str, tracer: Tracer, approver: Approver) -> Any:
        tool = self.get(name)
        with tracer.span(f"tool:{name}", kind="tool", agent=agent, args=args) as span:
            validate(args, tool.schema)
            if tool.requires_approval:
                approved = approver(ApprovalRequest(agent=agent, tool=name, args=args))
                span.attrs["approved"] = approved
                if not approved:
                    raise ApprovalDenied(f"{name} was not approved")
            attempts = tool.retries + 1
            for attempt in range(1, attempts + 1):
                span.attrs["attempts"] = attempt
                try:
                    return self._run_with_timeout(tool, args)
                except (TransientToolError, FutureTimeout) as exc:
                    if attempt == attempts:
                        raise TransientToolError(f"{name} failed after {attempts} attempts: {exc!r}") from exc
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _run_with_timeout(tool: Tool, args: dict[str, Any]) -> Any:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(tool.fn, **args).result(timeout=tool.timeout_s)
        finally:
            pool.shutdown(wait=False)  # a hung worker thread cannot be killed; we stop waiting for it
