"""AgentRelay: a small, testable multi-agent orchestration framework."""
from .agents import Agent, default_agents
from .bootstrap import build_orchestrator
from .orchestrator import Orchestrator, PlanError, RunResult
from .tools import ApprovalDenied, Tool, ToolRegistry, auto_approve, deny_all

__all__ = ["Agent", "ApprovalDenied", "Orchestrator", "PlanError", "RunResult", "Tool", "ToolRegistry",
           "auto_approve", "build_orchestrator", "default_agents", "deny_all"]
__version__ = "0.1.0"
