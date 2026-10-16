import time

import pytest

from agentrelay.tools import (ApprovalDenied, Tool, ToolRegistry, ToolValidationError, TransientToolError,
                              auto_approve, deny_all, validate)
from agentrelay.tracing import Tracer

SCHEMA = {"type": "object", "required": ["n"], "additionalProperties": False,
          "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}}}


def call(reg, name, args, approver=auto_approve):
    return reg.call(name, args, agent="t", tracer=Tracer(), approver=approver)


def test_validation_rejects_bad_types_bounds_and_extras():
    for bad in ({}, {"n": "3"}, {"n": True}, {"n": 9}, {"n": 2, "x": 1}):
        with pytest.raises(ToolValidationError):
            validate(bad, SCHEMA)
    validate({"n": 3}, SCHEMA)


def test_transient_errors_are_retried_then_succeed():
    calls = {"n": 0}

    def flaky(n):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError("blip")
        return n

    reg = ToolRegistry(backoff_s=0)
    reg.register(Tool("flaky", "", SCHEMA, flaky, retries=2))
    assert call(reg, "flaky", {"n": 2}) == 2 and calls["n"] == 3


def test_retries_exhausted_raises():
    reg = ToolRegistry(backoff_s=0)
    reg.register(Tool("down", "", SCHEMA, lambda n: (_ for _ in ()).throw(TransientToolError("x")), retries=1))
    with pytest.raises(TransientToolError, match="2 attempts"):
        call(reg, "down", {"n": 1})


def test_timeout_counts_as_transient():
    reg = ToolRegistry(backoff_s=0)
    reg.register(Tool("slow", "", SCHEMA, lambda n: time.sleep(0.3), timeout_s=0.05, retries=0))
    with pytest.raises(TransientToolError):
        call(reg, "slow", {"n": 1})


def test_side_effecting_tool_needs_approval():
    ran = []
    reg = ToolRegistry()
    reg.register(Tool("danger", "", SCHEMA, lambda n: ran.append(n), requires_approval=True))
    with pytest.raises(ApprovalDenied):
        call(reg, "danger", {"n": 1}, deny_all)
    assert ran == []  # the function must never run when approval is denied
    call(reg, "danger", {"n": 1}, auto_approve)
    assert ran == [1]


def test_invalid_args_never_reach_the_function():
    ran = []
    reg = ToolRegistry()
    reg.register(Tool("t", "", SCHEMA, lambda n: ran.append(n)))
    with pytest.raises(ToolValidationError):
        call(reg, "t", {"n": 99})
    assert ran == []
