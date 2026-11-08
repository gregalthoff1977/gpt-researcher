"""Request-local model-call observation; no global per-run logger state."""
from contextvars import ContextVar

research_observer = ContextVar("research_observer", default=None)


class ResearchLimitReached(RuntimeError):
    pass
