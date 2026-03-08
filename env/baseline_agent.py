"""
Rule-based baseline agent for LifeOps.

This agent is intentionally simple and provides a benchmark "floor" policy.
It uses action masking to avoid obviously illegal actions.

Rules (as requested):
- Always accept meetings during 9am-6pm (when feasible).
- Always reject meetings before 9am.
- Always block 8pm-11pm for productivity (when there is remaining task work).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from env.actions import Action, ActionType, mask_illegal_actions


def choose_baseline_action(state: Dict[str, Any], valid_actions: Sequence[Action]) -> Action:
    """
    Choose a concrete action from the env-provided `valid_actions`.

    The env action space includes parameterized actions (reschedule times, focus
    start times). This policy selects among them deterministically.
    """

    masked = mask_illegal_actions(state, list(valid_actions))
    if not masked:
        raise RuntimeError("No valid actions available")

    req = state.get("current_request")
    if req is None:
        # No request: try to block productivity 20:00-23:00 if tasks remain.
        tasks = state.get("tasks", []) or []
        has_work = any(int(t.get("remaining_minutes", 0)) > 0 for t in tasks)
        if has_work:
            for a in masked:
                if a.action_type == ActionType.block_focus_time and int(a.new_start_min or 0) >= 20 * 60:
                    return a
        # Otherwise pick any legal focus block, else first legal action.
        for a in masked:
            if a.action_type == ActionType.block_focus_time:
                return a
        return masked[0]

    start = int(req.get("start_min", 0))

    # Reject meetings before 9am.
    if start < 9 * 60:
        rej = next((a for a in masked if a.action_type == ActionType.reject_event), None)
        if rej is not None:
            return rej

    # Accept meetings during 9am-6pm when feasible.
    if 9 * 60 <= start < 18 * 60:
        acc = next((a for a in masked if a.action_type == ActionType.accept_event), None)
        if acc is not None:
            return acc

    # Otherwise try to reschedule to a feasible time (mask already ensured feasibility).
    res = next((a for a in masked if a.action_type == ActionType.reschedule_event), None)
    if res is not None:
        return res
    prop = next((a for a in masked if a.action_type == ActionType.propose_new_time), None)
    if prop is not None:
        return prop

    # Fallback: reject (still legal).
    rej = next((a for a in masked if a.action_type == ActionType.reject_event), None)
    return rej if rej is not None else masked[0]

