"""
Action space for LifeOps.

We keep actions structured (no natural language) and constrained by the current
state (e.g., you can only accept/reject the current incoming request).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActionType(str, Enum):
    accept_event = "accept_event"
    reject_event = "reject_event"
    reschedule_event = "reschedule_event"
    propose_new_time = "propose_new_time"
    block_focus_time = "block_focus_time"


@dataclass(frozen=True)
class Action:
    """
    A structured action.

    For request-handling actions, `request_id` is required.
    For time-changing actions, set `new_start_min` (and optionally `new_end_min`).
    For focus blocks, set `new_start_min` and `duration_min`.
    """

    action_type: ActionType
    request_id: Optional[str] = None
    new_start_min: Optional[int] = None
    new_end_min: Optional[int] = None
    duration_min: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "request_id": self.request_id,
            "new_start_min": self.new_start_min,
            "new_end_min": self.new_end_min,
            "duration_min": self.duration_min,
        }

    def key(self) -> Tuple[Any, ...]:
        """Stable tuple representation used for membership checks."""
        return (
            self.action_type.value,
            self.request_id,
            self.new_start_min,
            self.new_end_min,
            self.duration_min,
        )


def generate_valid_actions(state: Dict[str, Any]) -> List[Action]:
    """
    Generate a minimal, constrained action set from a JSON-like state.

    Assumptions:
    - The env surfaces a single "current_request" to handle next.
    - Focus blocks can be proposed only if there is at least one unfinished task.
    """

    actions: List[Action] = []
    current_req = state.get("current_request")
    if current_req is not None:
        req_id = current_req["event_id"]
        actions.extend(
            [
                Action(ActionType.accept_event, request_id=req_id),
                Action(ActionType.reject_event, request_id=req_id),
            ]
        )

        # Minimal reschedule/propose: shift by +/- 30 minutes, same duration.
        start = int(current_req["start_min"])
        end = int(current_req["end_min"])
        duration = max(0, end - start)
        for delta in (-30, 30, 60):
            new_start = max(0, min(1440 - duration, start + delta))
            actions.append(
                Action(
                    ActionType.reschedule_event,
                    request_id=req_id,
                    new_start_min=new_start,
                    new_end_min=new_start + duration,
                )
            )
            actions.append(
                Action(
                    ActionType.propose_new_time,
                    request_id=req_id,
                    new_start_min=new_start,
                    new_end_min=new_start + duration,
                )
            )

    # Focus blocks: propose a couple of common durations at common times.
    # Filter out any that would overlap with existing calendar events (same check as
    # reward.py: a_start < b_end and b_start < a_end) to avoid suggesting invalid
    # actions that would be penalized.
    tasks = state.get("tasks", [])
    has_unfinished = any(int(t.get("remaining_minutes", 0)) > 0 for t in tasks)
    calendar = state.get("calendar", [])
    if has_unfinished:
        for start_min in (9 * 60, 11 * 60, 13 * 60, 15 * 60, 17 * 60):
            for duration in (60,):
                focus_start = start_min
                focus_end = start_min + duration
                overlaps = any(
                    focus_start < int(e.get("end_min", 0)) and int(e.get("start_min", 0)) < focus_end
                    for e in calendar
                )
                if not overlaps:
                    actions.append(
                        Action(
                            ActionType.block_focus_time,
                            request_id=None,
                            new_start_min=start_min,
                            duration_min=duration,
                        )
                    )

    return actions

