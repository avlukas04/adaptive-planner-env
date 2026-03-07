"""
Reward calculation for LifeOps.

The reward is intentionally small and readable. It's "shaped" to encourage:
- feasible schedules (no overlaps, feasible travel)
- respecting persona preferences (meeting windows)
- making progress on tasks via focus blocks
- handling important requests appropriately
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def detect_overlaps(events: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Returns pairs of event_ids that overlap in time.
    """

    overlaps: List[Tuple[str, str]] = []
    n = len(events)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = events[i], events[j]
            if _overlap(int(a["start_min"]), int(a["end_min"]), int(b["start_min"]), int(b["end_min"])):
                overlaps.append((str(a["event_id"]), str(b["event_id"])))
    return overlaps


def _travel_time_minutes(travel_times: Dict[str, Dict[str, int]], a_loc: str, b_loc: str) -> int:
    """
    Fetch travel time. If unknown, default conservatively:
    - same location: 0
    - otherwise: 30
    """

    if a_loc == b_loc:
        return 0
    return int(travel_times.get(a_loc, {}).get(b_loc, 30))


def travel_issues(
    events: List[Dict[str, Any]],
    travel_times: Dict[str, Dict[str, int]],
) -> List[Tuple[str, str, int, int]]:
    """
    Returns travel feasibility issues between consecutive events.

    Output tuple: (from_event_id, to_event_id, needed_minutes, available_minutes)
    """

    if not events:
        return []

    ordered = sorted(events, key=lambda e: (int(e["start_min"]), int(e["end_min"])))
    issues: List[Tuple[str, str, int, int]] = []
    for prev, nxt in zip(ordered, ordered[1:]):
        prev_end = int(prev["end_min"])
        nxt_start = int(nxt["start_min"])
        available = nxt_start - prev_end
        # If events overlap, that infeasibility is handled by overlap penalties.
        # Avoid double-penalizing with travel constraints.
        if available < 0:
            continue
        needed = _travel_time_minutes(travel_times, str(prev["location"]), str(nxt["location"]))
        if needed > available:
            issues.append((str(prev["event_id"]), str(nxt["event_id"]), needed, available))
    return issues


def compute_reward(
    prev_state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward from previous->next state transition.

    Returns (reward, reward_breakdown_dict).
    """

    breakdown: Dict[str, Any] = {}
    reward = 0.0

    prev_events = list(prev_state.get("calendar", []))
    next_events = list(next_state.get("calendar", []))
    persona: Dict[str, Any] = next_state["persona"]
    travel_times = next_state.get("travel_times", {})

    prev_overlaps = detect_overlaps(prev_events)
    next_overlaps = detect_overlaps(next_events)
    if next_overlaps:
        reward -= 5.0 * len(next_overlaps)
        breakdown["overlap_penalty"] = -5.0 * len(next_overlaps)
    if prev_overlaps and len(next_overlaps) < len(prev_overlaps):
        reward += 3.0
        breakdown["conflict_resolved_bonus"] = 3.0

    issues = travel_issues(next_events, travel_times)
    if issues:
        # Penalize per infeasible leg.
        travel_pen = -4.0 * len(issues) * float(persona.get("travel_aversion_weight", 1.0))
        reward += travel_pen
        breakdown["travel_penalty"] = travel_pen
        breakdown["travel_issues"] = issues

    # Penalize rejecting important requests.
    if action.get("action_type") == "reject_event" and next_state.get("last_handled_request"):
        handled = next_state["last_handled_request"]
        if int(handled.get("importance", 1)) >= 3:
            reward -= 4.0
            breakdown["rejected_important_penalty"] = -4.0

    # Persona preference: meetings outside preferred window.
    # We approximate: if the action added a new event (accept/reschedule/propose),
    # check the last added event time against persona preference window.
    if action.get("action_type") in {"accept_event", "reschedule_event", "propose_new_time"}:
        added = next_state.get("last_added_event")
        if added is not None and added.get("kind", "meeting") in {"meeting", "obligation", "personal"}:
            pen_units = _meeting_window_penalty(persona, int(added["start_min"]), int(added["end_min"]))
            if pen_units > 0:
                pref_pen = -2.0 * pen_units
                reward += pref_pen
                breakdown["preference_penalty"] = pref_pen

    # Reward task progress from focus blocks.
    if action.get("action_type") == "block_focus_time":
        progress = int(next_state.get("last_task_progress_minutes", 0))
        if progress > 0:
            focus_rew = (1.0 + 0.02 * progress) * float(persona.get("focus_time_weight", 1.0))
            reward += focus_rew
            breakdown["focus_reward"] = focus_rew
        else:
            reward -= 0.5  # wasted focus block
            breakdown["wasted_focus_penalty"] = -0.5

    breakdown["total"] = reward
    return reward, breakdown


def _meeting_window_penalty(persona: Dict[str, Any], start_min: int, end_min: int) -> float:
    """
    Persona preference penalty (non-negative).

    Persona dict schema (from env):
    - preferred_meeting_window: [start, end]
    - avoid_meetings_before_min: int|None
    - avoid_meetings_after_min: int|None
    - preference_weight: float
    """

    pref = persona.get("preferred_meeting_window", [0, 1440])
    pref_start, pref_end = int(pref[0]), int(pref[1])
    avoid_before = persona.get("avoid_meetings_before_min")
    avoid_after = persona.get("avoid_meetings_after_min")
    weight = float(persona.get("preference_weight", 1.0))

    penalty_units = 0.0
    if start_min < pref_start or end_min > pref_end:
        penalty_units += 1.0
    if avoid_before is not None and start_min < int(avoid_before):
        penalty_units += 1.0
    if avoid_after is not None and end_min > int(avoid_after):
        penalty_units += 1.0
    return penalty_units * weight

