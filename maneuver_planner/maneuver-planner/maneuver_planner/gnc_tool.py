"""The small interface used by Level 0 tools; no registry or planner is required."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from .models import ManeuverCandidate

RequestT = TypeVar("RequestT", contravariant=True)


@dataclass(frozen=True)
class Applicability:
    applicable: bool
    reason: str


@runtime_checkable
class GncTool(Protocol[RequestT]):
    name: str
    description: str

    def check_applicability(self, request: RequestT) -> Applicability:
        """State whether this tool's mathematical assumptions fit the request."""
        ...

    def propose(self, request: RequestT) -> ManeuverCandidate:
        """Return proposed commands without executing them or modifying the request."""
        ...
