"""
Reward calculation for LifeOps.

The reward is intentionally small and readable. It's "shaped" to encourage:
- feasible schedules (no overlaps, feasible travel)
- respecting persona preferences (meeting windows)
- making progress on tasks via focus blocks
- handling important requests appropriately
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
    start_location: Optional[str] = None,
) -> List[Tuple[str, str, int, int]]:
    """
    Returns travel feasibility issues between consecutive events.

    If start_location is provided (e.g. persona home), also checks whether the
    user can reach the first event of the day in time.

    Output tuple: (from_event_id, to_event_id, needed_minutes, available_minutes)
    """

    if not events:
        return []

    ordered = sorted(events, key=lambda e: (int(e["start_min"]), int(e["end_min"])))
    issues: List[Tuple[str, str, int, int]] = []

    # Check travel from start_location to first event (if provided).
    if start_location is not None:
        first = ordered[0]
        available = int(first["start_min"])
        needed = _travel_time_minutes(travel_times, start_location, str(first["location"]))
        if needed > available:
            issues.append(("__start__", str(first["event_id"]), needed, available))

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
    action_type = str(action.get("action_type", ""))
    day_of_week = str(next_state.get("day_of_week", ""))

    prev_overlaps = detect_overlaps(prev_events)
    next_overlaps = detect_overlaps(next_events)
    if next_overlaps:
        # Penalty exists, but should not dominate the episode.
        overlap_pen = -3.0 * len(next_overlaps)
        reward += overlap_pen
        breakdown["overlap_penalty"] = overlap_pen
    breakdown["overlap_count"] = len(next_overlaps)

    issues = travel_issues(
        next_events,
        travel_times,
        start_location=persona.get("home_location"),
    )
    if issues:
        # Penalize per infeasible leg.
        travel_pen = -2.0 * len(issues) * float(persona.get("travel_aversion_weight", 1.0))
        reward += travel_pen
        breakdown["travel_penalty"] = travel_pen
        breakdown["travel_issues"] = issues
    breakdown["travel_issue_count"] = len(issues)

    # Small positive signal for "making a decision" on an incoming request.
    if action_type in {"accept_event", "reject_event", "reschedule_event", "propose_new_time"}:
        reward += 0.5
        breakdown["handled_request_bonus"] = 0.5

    # Penalize rejecting important requests.
    if action_type == "reject_event" and next_state.get("last_handled_request"):
        handled = next_state["last_handled_request"]
        if int(handled.get("importance", 1)) >= 3:
            reward -= 3.0
            breakdown["rejected_important_penalty"] = -3.0
        else:
            # If rejecting a low-importance request avoids infeasibility, that's fine.
            reward += 0.5
            breakdown["rejected_low_importance_bonus"] = 0.5

    # Persona preference: meetings outside preferred window.
    # We approximate: if the action added a new event (accept/reschedule/propose),
    # check the last added event time against persona preference window.
    if action_type in {"accept_event", "reschedule_event", "propose_new_time"}:
        added = next_state.get("last_added_event")
        if added is not None and added.get("kind", "meeting") in {"meeting", "obligation", "personal"}:
            pen_units = _meeting_window_penalty(persona, int(added["start_min"]), int(added["end_min"]))
            if pen_units > 0:
                # Preference violations should matter, but not swamp feasibility.
                pref_pen = -1.5 * pen_units
                reward += pref_pen
                breakdown["preference_penalty"] = pref_pen
            else:
                reward += 1.0
                breakdown["preference_bonus"] = 1.0

    # Reward task progress from focus blocks.
    if action_type == "block_focus_time":
        progress = int(next_state.get("last_task_progress_minutes", 0))
        if progress > 0:
            # Make focus explicitly valuable so an agent can earn positive reward.
            # Typical focus block should be +3..+5.
            focus_rew = (3.0 + 0.02 * progress) * float(persona.get("focus_time_weight", 1.0))
            reward += focus_rew
            breakdown["focus_reward"] = focus_rew
        else:
            reward -= 0.5  # wasted focus block
            breakdown["wasted_focus_penalty"] = -0.5

    # Additional positive shaping for feasible scheduling choices.
    # If the action scheduled something (accept/reschedule), and the resulting
    # calendar is feasible, reward it more strongly.
    if action_type in {"accept_event", "reschedule_event"} and next_state.get("last_added_event") is not None:
        if not next_overlaps and not issues:
            # Strong, clearly positive signal. Scale with importance so that
            # low-importance accepts don't dominate decisions.
            handled_req = next_state.get("last_handled_request") or {}
            imp = int(handled_req.get("importance", 1))
            bonus = 4.0 if imp >= 2 else 2.0
            reward += bonus
            breakdown["feasible_schedule_bonus"] = bonus

    # Propose_new_time doesn't schedule, but suggesting a feasible alternative
    # is still useful (smaller reward than actually scheduling).
    if action_type == "propose_new_time" and next_state.get("last_added_event") is not None:
        # Treat as "good" if the suggested slot doesn't overlap existing events
        # and doesn't introduce travel impossibilities if it were scheduled.
        suggested = next_state["last_added_event"]
        sim_events = list(next_events) + [suggested]
        if not detect_overlaps(sim_events) and not travel_issues(sim_events, travel_times, start_location=persona.get("home_location")):
            reward += 2.0
            breakdown["feasible_proposal_bonus"] = 2.0

    # Edge-case shaping: flexible=False requests should not be rescheduled lightly.
    handled_req = next_state.get("last_handled_request") or {}
    if action_type in {"reschedule_event", "propose_new_time"} and handled_req:
        if bool(handled_req.get("flexible", True)) is False:
            reward -= 5.0
            breakdown["inflexible_reschedule_penalty"] = -5.0

    # Cascading conflict shaping: if we accept a low-importance meeting that overlaps a
    # future inflexible important request (preview), penalize the choice.
    if action_type == "accept_event" and handled_req:
        imp = int(handled_req.get("importance", 1))
        if imp <= 1:
            preview = next_state.get("upcoming_requests_preview") or []
            a_s = int(handled_req.get("start_min", 0))
            a_e = int(handled_req.get("end_min", 0))
            for r in preview:
                if int(r.get("importance", 1)) >= 3 and bool(r.get("flexible", True)) is False:
                    r_s = int(r.get("start_min", 0))
                    r_e = int(r.get("end_min", 0))
                    if a_s < r_e and r_s < a_e:
                        reward -= 3.0
                        breakdown["cascade_penalty"] = -3.0
                        break

    # Edge-case shaping: Monday mood pattern (cancels low-importance meetings).
    # If it's Monday and we accept a low-importance meeting, penalize.
    if day_of_week.lower() == "monday" and action_type == "accept_event" and handled_req:
        if int(handled_req.get("importance", 1)) <= 2:
            reward -= 4.0
            breakdown["monday_cancellation_penalty"] = -4.0

    # Edge-case shaping: deadline pressure.
    # If there is a task due today/tomorrow and still lots of remaining work,
    # apply a small per-step penalty to encourage focus blocks earlier.
    tasks_next = next_state.get("tasks", []) or []
    urgent_remaining = 0
    for t in tasks_next:
        due = t.get("due_in_days")
        if due is None:
            continue
        if int(due) <= 1:
            urgent_remaining += int(t.get("remaining_minutes", 0))
    if urgent_remaining > 0:
        # Scale gently; shouldn't dominate but should steer decisions.
        pen = -min(2.0, 0.01 * float(urgent_remaining))
        reward += pen
        breakdown["deadline_pressure_penalty"] = pen

    # Truncation penalty: if the episode ended due to step limit with goals unfinished,
    # apply a one-time penalty to discourage "accept everything" behavior under deadlines.
    if bool(next_state.get("_truncated", False)):
        remaining = sum(int(t.get("remaining_minutes", 0)) for t in tasks_next)
        if remaining > 0:
            reward -= 4.0
            breakdown["truncation_goal_miss_penalty"] = -4.0

    # Task progress / goal shaping: reward reducing remaining minutes.
    prev_remaining = sum(int(t.get("remaining_minutes", 0)) for t in prev_state.get("tasks", []) or [])
    next_remaining = sum(int(t.get("remaining_minutes", 0)) for t in next_state.get("tasks", []) or [])
    if next_remaining < prev_remaining:
        # Small extra reward for making progress (in addition to focus reward).
        prog = prev_remaining - next_remaining
        bonus = min(2.0, 0.01 * float(prog))
        reward += bonus
        breakdown["task_progress_bonus"] = bonus

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

