"""
Episode tracing and structured logging for LifeOps.

Provides human-readable step-by-step logs and a timeline view for hackathon demos.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _min_to_time(m: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


@dataclass
class StepRecord:
    """Record of a single environment step."""

    step: int
    action: Dict[str, Any]
    prev_calendar_count: int
    next_calendar_count: int
    prev_pending_count: int
    next_pending_count: int
    reward: float
    breakdown: Dict[str, Any]
    overlaps: List[tuple]
    travel_issues: List[tuple]
    done: bool

    # State changes (compact summaries)
    added_event: Optional[Dict[str, Any]] = None
    handled_request: Optional[Dict[str, Any]] = None
    task_progress: Optional[Dict[str, str]] = None  # task_id -> "X min progress"


@dataclass
class EpisodeTrace:
    """Trace of an entire episode for logging and timeline display."""

    scenario_id: str
    persona_name: str
    initial_calendar: List[Dict[str, Any]] = field(default_factory=list)
    initial_tasks: List[Dict[str, Any]] = field(default_factory=list)
    initial_pending_count: int = 0
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0

    def log_step(
        self,
        step: int,
        action: Dict[str, Any],
        prev_obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        reward: float,
        breakdown: Dict[str, Any],
        info: Dict[str, Any],
        done: bool,
        last_added_event: Optional[Dict[str, Any]] = None,
        last_handled_request: Optional[Dict[str, Any]] = None,
        last_task_progress_minutes: int = 0,
        task_id_progressed: Optional[str] = None,
    ) -> None:
        """Record one step."""
        task_progress = None
        if last_task_progress_minutes and task_id_progressed:
            task_progress = {task_id_progressed: f"{last_task_progress_minutes} min"}

        self.steps.append(
            StepRecord(
                step=step,
                action=copy.deepcopy(action),
                prev_calendar_count=len(prev_obs.get("calendar", [])),
                next_calendar_count=len(next_obs.get("calendar", [])),
                prev_pending_count=prev_obs.get("pending_request_count", 0),
                next_pending_count=next_obs.get("pending_request_count", 0),
                reward=reward,
                breakdown=copy.deepcopy(breakdown),
                overlaps=info.get("overlaps", []),
                travel_issues=info.get("travel_issues", []),
                done=done,
                added_event=copy.deepcopy(last_added_event) if last_added_event else None,
                handled_request=copy.deepcopy(last_handled_request) if last_handled_request else None,
                task_progress=task_progress,
            )
        )

    def _format_action(self, a: Dict[str, Any]) -> str:
        at = a.get("action_type", "?")
        if at == "block_focus_time":
            start = a.get("new_start_min")
            dur = a.get("duration_min")
            return f"block_focus_time @ {_min_to_time(start or 0)} for {dur} min"
        if at == "accept_event":
            return f"accept_event (request_id={a.get('request_id', '?')})"
        if at == "reject_event":
            return f"reject_event (request_id={a.get('request_id', '?')})"
        if at == "reschedule_event":
            ns, ne = a.get("new_start_min"), a.get("new_end_min")
            return f"reschedule_event → {_min_to_time(ns or 0)}–{_min_to_time(ne or 0)}"
        if at == "propose_new_time":
            ns, ne = a.get("new_start_min"), a.get("new_end_min")
            return f"propose_new_time → {_min_to_time(ns or 0)}–{_min_to_time(ne or 0)}"
        return str(a)

    def _format_breakdown(self, b: Dict[str, Any]) -> str:
        parts = []
        for k, v in b.items():
            if k == "total":
                continue
            if isinstance(v, (int, float)) and v != 0:
                parts.append(f"{k}={v:+.1f}")
        return ", ".join(parts) if parts else "(none)"

    def print_step_log(self, step_record: StepRecord) -> None:
        """Print a single step in human-readable form."""
        s = step_record
        print(f"\n  Step {s.step}")
        print(f"    Action: {self._format_action(s.action)}")
        print(f"    Reward: {s.reward:+.2f}  ({self._format_breakdown(s.breakdown)})")
        if s.added_event:
            e = s.added_event
            print(f"    + Added: {e.get('title', '?')} @ {_min_to_time(e.get('start_min', 0))}–{_min_to_time(e.get('end_min', 0))} ({e.get('location', '?')})")
        if s.handled_request and s.action.get("action_type") != "block_focus_time":
            r = s.handled_request
            at = s.action.get("action_type", "")
            if at == "reject_event":
                outcome = "rejected"
            elif at == "propose_new_time":
                outcome = "proposed new time (not scheduled)"
            else:
                outcome = "accepted/scheduled"
            print(f"    Request {outcome}: {r.get('title', '?')}")
        if s.task_progress:
            for tid, prog in s.task_progress.items():
                print(f"    Task progress: {tid} ({prog})")
        if s.overlaps:
            print(f"    ⚠ Overlaps: {s.overlaps}")
        if s.travel_issues:
            print(f"    ⚠ Travel issues: {[(t[0], t[1], f'need {t[2]}min') for t in s.travel_issues]}")

    def print_timeline(self, final_calendar: Optional[List[Dict[str, Any]]] = None) -> None:
        """Print a readable timeline of the final calendar."""
        if final_calendar is not None:
            events = list(final_calendar)
        else:
            # Fallback: merge initial + all added events from steps
            events = list(self.initial_calendar)
            for s in self.steps:
                if s.added_event:
                    events.append(s.added_event)

        if not events:
            print("\n  (No events on calendar)")
            return

        ordered = sorted(events, key=lambda e: (int(e["start_min"]), int(e["end_min"])))
        print("\n  Timeline (final calendar):")
        print("  " + "-" * 60)
        for e in ordered:
            start = int(e["start_min"])
            end = int(e["end_min"])
            title = e.get("title", e.get("event_id", "?"))
            loc = e.get("location", "?")
            kind = e.get("kind", "meeting")
            print(f"  {_min_to_time(start)} – {_min_to_time(end)}  {title}  @ {loc}  [{kind}]")
        print("  " + "-" * 60)

    def print_full(self, final_calendar: Optional[List[Dict[str, Any]]] = None) -> None:
        """Print the complete episode trace (header, steps, timeline, summary)."""
        print("\n" + "=" * 60)
        print("EPISODE TRACE")
        print("=" * 60)
        print(f"Scenario: {self.scenario_id}")
        print(f"Persona: {self.persona_name}")
        print(f"Initial: {len(self.initial_calendar)} events, {len(self.initial_tasks)} tasks, {self.initial_pending_count} pending requests")
        print("-" * 60)

        for s in self.steps:
            self.print_step_log(s)

        self.print_timeline(final_calendar)

        print("\n" + "-" * 60)
        print(f"Total reward: {self.total_reward:+.2f}")
        print("=" * 60)
