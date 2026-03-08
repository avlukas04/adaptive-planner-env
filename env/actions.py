"""
Action space for LifeOps.

We keep actions structured (no natural language) and constrained by the current
state (e.g., you can only accept/reject the current incoming request).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    tasks = state.get("tasks", [])
    has_unfinished = any(int(t.get("remaining_minutes", 0)) > 0 for t in tasks)
    if has_unfinished:
        for start_min in (9 * 60, 11 * 60, 14 * 60, 16 * 60):
            for duration in (30, 60):
                actions.append(
                    Action(
                        ActionType.block_focus_time,
                        request_id=None,
                        new_start_min=start_min,
                        duration_min=duration,
                    )
                )

    return actions


def mask_illegal_actions(
    state: Dict[str, Any],
    actions: Sequence[Action],
    *,
    blocked_meeting_hours: Sequence[Tuple[int, int]] = ((0, 9 * 60), (20 * 60, 23 * 60)),
) -> List[Action]:
    """
    Filter out actions that would be illegal or obviously infeasible.

    This is a *policy-side* action mask applied before any agent samples.
    It reduces variance and avoids repeatedly taking actions that guarantee
    negative reward (double-booking, impossible travel, or scheduling meetings
    inside blocked hours).

    Mask rules:
    - **Double booking**: scheduled event overlaps an existing calendar event.
    - **Travel violations**: scheduled event makes consecutive travel impossible.
    - **Blocked hours**: meeting-like requests cannot be scheduled in blocked hours.

    Notes:
    - `propose_new_time` does not schedule in env semantics, but proposing a
      blocked-hour time is still treated as illegal to avoid bad suggestions.
    - Focus blocks are allowed in blocked hours, but are still masked if they
      overlap existing events or create travel issues.
    """
    from env.reward import detect_overlaps, travel_issues

    calendar = list(state.get("calendar", []))
    req = state.get("current_request") or None
    persona = state.get("persona", {}) or {}
    travel_times = state.get("travel_times", {}) or {}

    def overlaps_blocked(start_min: int, end_min: int) -> bool:
        for bs, be in blocked_meeting_hours:
            if start_min < be and bs < end_min:
                return True
        return False

    masked: List[Action] = []
    for a in actions:
        at = a.action_type.value
        if at == ActionType.reject_event.value:
            masked.append(a)
            continue

        if at == ActionType.block_focus_time.value:
            start = int(a.new_start_min or 0)
            dur = int(a.duration_min or 0)
            end = start + dur
            focus_event = {
                "event_id": "__focus__",
                "start_min": start,
                "end_min": end,
                "location": persona.get("primary_work_location", "Home"),
                "kind": "focus",
                "importance": 1,
            }
            sim = calendar + [focus_event]
            if detect_overlaps(sim):
                continue
            if travel_issues(sim, travel_times, start_location=persona.get("home_location")):
                continue
            masked.append(a)
            continue

        # Request-handling actions require a request.
        if req is None:
            continue

        # Determine scheduled/suggested time.
        if at in {ActionType.accept_event.value}:
            start = int(req.get("start_min", 0))
            end = int(req.get("end_min", 0))
        elif at in {ActionType.reschedule_event.value, ActionType.propose_new_time.value}:
            start = int(a.new_start_min or req.get("start_min", 0))
            end = int(a.new_end_min or req.get("end_min", 0))
        else:
            # Unknown action type; drop.
            continue

        # Meetings should not be scheduled (or proposed) during blocked hours.
        kind = str(req.get("kind", "meeting"))
        if kind in {"meeting", "obligation", "personal"} and overlaps_blocked(start, end):
            continue

        candidate = dict(req)
        candidate["start_min"] = start
        candidate["end_min"] = end

        # For proposals, check feasibility as if it were scheduled.
        sim = calendar + [candidate]
        if detect_overlaps(sim):
            continue
        if travel_issues(sim, travel_times, start_location=persona.get("home_location")):
            continue

        masked.append(a)

    # Always return something if possible; if we masked everything, fall back to original.
    return masked if masked else list(actions)

