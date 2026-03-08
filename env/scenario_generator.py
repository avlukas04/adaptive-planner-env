"""
Scenario generation for LifeOps (MVP).

We provide a fixed set of sample scenarios (>=5) to keep things deterministic and
hackathon-friendly. Scenarios include:
- existing calendar events
- tasks/goals
- incoming requests (to be handled by the agent)
- a travel-time matrix for feasibility checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from env.personas import h2m


@dataclass(frozen=True)
class Event:
    event_id: str
    title: str
    start_min: int
    end_min: int
    location: str
    importance: int = 1  # 1=low, 2=medium, 3=high
    kind: str = "meeting"  # meeting|obligation|focus|personal

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_min": self.start_min,
            "end_min": self.end_min,
            "location": self.location,
            "importance": self.importance,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    remaining_minutes: int
    priority: int = 2  # 1=low, 2=medium, 3=high
    due_in_days: Optional[int] = None  # 0=today, 1=tomorrow, etc. (for deadline pressure)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "remaining_minutes": self.remaining_minutes,
            "priority": self.priority,
            "due_in_days": self.due_in_days,
        }


@dataclass(frozen=True)
class IncomingRequest(Event):
    """
    An incoming scheduling request.

    `flexible` controls whether the agent can reschedule without heavy penalty.
    """

    from_person: str = "Unknown"
    flexible: bool = True

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({"from_person": self.from_person, "flexible": self.flexible})
        return d


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    persona_id: str
    calendar: List[Event]
    tasks: List[Task]
    incoming_requests: List[IncomingRequest]
    travel_times: Dict[str, Dict[str, int]]
    # Optional metadata to enable "edge case" behavior.
    day_of_week: str = "Wednesday"  # Monday..Sunday
    tags: Optional[List[str]] = None


def default_travel_times() -> Dict[str, Dict[str, int]]:
    """
    Simple travel times (minutes) between common locations.
    Missing entries are treated as "unknown" by the env and defaulted conservatively.
    """

    locs = ["Home", "Office", "Downtown", "Gym", "School"]
    base: Dict[str, Dict[str, int]] = {a: {b: (0 if a == b else 30) for b in locs} for a in locs}

    # Some more specific travel times.
    base["Home"]["Office"] = 25
    base["Office"]["Home"] = 25
    base["Home"]["School"] = 12
    base["School"]["Home"] = 12
    base["Office"]["Downtown"] = 18
    base["Downtown"]["Office"] = 18
    base["Home"]["Downtown"] = 35
    base["Downtown"]["Home"] = 35
    base["Office"]["Gym"] = 10
    base["Gym"]["Office"] = 10
    base["Home"]["Gym"] = 20
    base["Gym"]["Home"] = 20
    base["School"]["Office"] = 28
    base["Office"]["School"] = 28
    return base


def sample_scenarios() -> List[Scenario]:
    """
    Requirement: at least 5 sample scenarios.
    """

    tt = default_travel_times()
    return [
        # -------------------------
        # EASY curriculum scenarios
        # -------------------------
        # Easy: 2-3 events, no travel conflicts, no goals/tasks.
        Scenario(
            scenario_id="easy_1",
            name="Easy: simple accept",
            persona_id="early_bird_engineer",
            calendar=[
                Event("e1", "Morning check-in", h2m(10, 0), h2m(10, 30), "Office", importance=2),
                Event("e2", "Lunch", h2m(12, 30), h2m(13, 0), "Office", importance=1, kind="personal"),
            ],
            tasks=[],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Quick 1:1",
                    h2m(11, 0),
                    h2m(11, 30),
                    "Office",
                    importance=2,
                    from_person="Teammate",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Thursday",
            tags=["easy"],
        ),
        Scenario(
            scenario_id="easy_2",
            name="Easy: simple reject (too early)",
            persona_id="busy_parent",
            calendar=[
                Event("e1", "Morning routine", h2m(9, 30), h2m(10, 30), "Home", importance=2, kind="personal"),
                Event("e2", "Work block", h2m(10, 45), h2m(12, 0), "Home", importance=2, kind="focus"),
            ],
            tasks=[],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Early call",
                    h2m(8, 15),
                    h2m(8, 30),
                    "Home",
                    importance=1,
                    from_person="Acquaintance",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Tuesday",
            tags=["easy"],
        ),

        # ---------------------------
        # MEDIUM curriculum scenarios
        # ---------------------------
        # Medium: 4-5 events, 1 travel constraint, 1 goal/task.
        Scenario(
            scenario_id="medium_1",
            name="Medium: one travel constraint + one goal",
            persona_id="night_owl_creator",
            calendar=[
                Event("e1", "Coffee", h2m(11, 0), h2m(11, 30), "Downtown", importance=1, kind="personal"),
                Event("e2", "Work session", h2m(12, 0), h2m(13, 0), "Downtown", importance=2, kind="focus"),
                Event("e3", "Call with client", h2m(15, 0), h2m(15, 30), "Office", importance=2),
                Event("e4", "Gym", h2m(18, 0), h2m(19, 0), "Gym", importance=1, kind="personal"),
            ],
            tasks=[Task("t1", "Edit portfolio", remaining_minutes=60, priority=2, due_in_days=3)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Lunch meeting",
                    h2m(13, 10),
                    h2m(13, 40),
                    "Office",
                    importance=2,
                    from_person="Friend",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Wednesday",
            tags=["medium"],
        ),
        Scenario(
            scenario_id="medium_2",
            name="Medium: important obligation protected",
            persona_id="busy_parent",
            calendar=[
                Event("e1", "Work block", h2m(9, 0), h2m(11, 0), "Office", importance=2, kind="focus"),
                Event("e2", "Doctor appointment", h2m(12, 0), h2m(12, 30), "Downtown", importance=3, kind="obligation"),
                Event("e3", "Work block", h2m(13, 0), h2m(14, 30), "Office", importance=2, kind="focus"),
                Event("e4", "School pickup", h2m(15, 30), h2m(16, 0), "School", importance=3, kind="obligation"),
            ],
            tasks=[Task("t1", "Prepare report", remaining_minutes=45, priority=3, due_in_days=2)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Team sync",
                    h2m(12, 15),
                    h2m(12, 45),
                    "Office",
                    importance=2,
                    from_person="Team",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Friday",
            tags=["medium"],
        ),

        # -------------------------
        # EDGE CASE scenarios (hard)
        # -------------------------
        # These are designed so the fixed-rule baseline tends to do worse than
        # an LLM that can look at patterns, upcoming requests, and deadlines.
        Scenario(
            scenario_id="edge_cascade_1",
            name="Edge: cascading conflicts (protect inflexible obligations)",
            persona_id="early_bird_engineer",
            calendar=[
                Event("e1", "Deep work", h2m(9, 0), h2m(10, 0), "Office", importance=2, kind="focus"),
                Event("e2", "Team update", h2m(12, 0), h2m(12, 30), "Office", importance=2),
                Event("e3", "Wrap up", h2m(17, 0), h2m(17, 30), "Office", importance=1, kind="personal"),
            ],
            tasks=[Task("t1", "Write spec", remaining_minutes=60, priority=3, due_in_days=1)],
            incoming_requests=[
                # Current (low) request overlaps the first inflexible request.
                IncomingRequest(
                    "r_low",
                    "Optional vendor intro",
                    h2m(10, 0),
                    h2m(10, 30),
                    "Office",
                    importance=1,
                    from_person="Vendor",
                    flexible=True,
                ),
                IncomingRequest(
                    "r_imp1",
                    "Legal review (inflexible)",
                    h2m(10, 0),
                    h2m(10, 30),
                    "Office",
                    importance=3,
                    from_person="Legal",
                    flexible=False,
                ),
                IncomingRequest(
                    "r_imp2",
                    "Security sign-off (inflexible)",
                    h2m(10, 30),
                    h2m(11, 0),
                    "Office",
                    importance=3,
                    from_person="Security",
                    flexible=False,
                ),
            ],
            travel_times=tt,
            day_of_week="Wednesday",
            tags=["edge", "cascade"],
        ),
        Scenario(
            scenario_id="edge_mood_monday_1",
            name="Edge: Monday mood (historically cancels)",
            persona_id="night_owl_creator",
            calendar=[
                Event("e1", "Creative block", h2m(11, 0), h2m(12, 30), "Home", importance=2, kind="focus"),
                Event("e2", "Short errands", h2m(16, 0), h2m(16, 30), "Downtown", importance=1, kind="personal"),
            ],
            tasks=[Task("t1", "Ship draft", remaining_minutes=60, priority=2, due_in_days=2)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Casual sync",
                    h2m(12, 30),
                    h2m(13, 0),
                    "Home",
                    importance=1,
                    from_person="Friend",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Monday",
            tags=["edge", "mood_monday"],
        ),
        Scenario(
            scenario_id="edge_deadline_1",
            name="Edge: goal deadline pressure (due today)",
            persona_id="busy_parent",
            calendar=[
                Event("e1", "School drop-off", h2m(8, 30), h2m(9, 0), "School", importance=3, kind="obligation"),
                # Intentionally "messy": stacked focus blocks + little slack. This makes the incoming
                # request conflict obvious in the demo UI.
                Event("e2", "Work block", h2m(9, 30), h2m(11, 0), "Office", importance=2, kind="focus"),
                Event("e4", "Pitch deck focus block", h2m(11, 0), h2m(12, 0), "Office", importance=3, kind="focus"),
                Event("e5", "Email triage", h2m(12, 30), h2m(13, 0), "Office", importance=1, kind="personal"),
                Event("e6", "Prep notes", h2m(13, 30), h2m(14, 0), "Office", importance=2, kind="focus"),
                Event("e3", "School pickup", h2m(15, 30), h2m(16, 0), "School", importance=3, kind="obligation"),
            ],
            tasks=[
                Task("goal1", "Startup pitch deck (DUE TODAY)", remaining_minutes=180, priority=3, due_in_days=0),
            ],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Status meeting",
                    h2m(11, 0),
                    h2m(11, 30),
                    "Office",
                    importance=2,
                    from_person="Manager",
                    flexible=True,
                ),
                IncomingRequest(
                    "r2",
                    "Client call",
                    h2m(13, 0),
                    h2m(13, 30),
                    "Office",
                    importance=2,
                    from_person="Client",
                    flexible=True,
                ),
                IncomingRequest(
                    "r3",
                    "Optional lunch",
                    h2m(12, 0),
                    h2m(12, 30),
                    "Office",
                    importance=1,
                    from_person="Colleague",
                    flexible=True,
                ),
            ],
            travel_times=tt,
            day_of_week="Thursday",
            tags=["edge", "deadline"],
        ),

        Scenario(
            scenario_id="s1_basic_conflict",
            name="Overlapping client sync",
            persona_id="early_bird_engineer",
            calendar=[
                Event("e1", "Daily standup", h2m(9, 30), h2m(10, 0), "Office", importance=2),
                Event("e2", "Code review block", h2m(10, 0), h2m(11, 0), "Office", importance=1, kind="focus"),
            ],
            tasks=[Task("t1", "Ship feature X", remaining_minutes=90, priority=3)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Client sync",
                    h2m(9, 45),
                    h2m(10, 15),
                    "Office",
                    importance=3,
                    from_person="Client PM",
                    flexible=False,
                )
            ],
            travel_times=tt,
            day_of_week="Wednesday",
            tags=["hard"],
        ),
        Scenario(
            scenario_id="s2_travel_tight",
            name="Tight travel between locations",
            persona_id="night_owl_creator",
            calendar=[
                Event("e1", "Brunch meeting", h2m(11, 30), h2m(12, 15), "Downtown", importance=2),
            ],
            tasks=[Task("t1", "Edit portfolio", remaining_minutes=60, priority=2)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Quick studio visit",
                    h2m(12, 20),
                    h2m(12, 50),
                    "Office",
                    importance=2,
                    from_person="Studio",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Tuesday",
            tags=["hard"],
        ),
        Scenario(
            scenario_id="s3_focus_vs_meeting",
            name="Focus time vs optional meeting",
            persona_id="early_bird_engineer",
            calendar=[
                Event("e1", "Architecture deep work", h2m(13, 0), h2m(15, 0), "Office", importance=2, kind="focus"),
            ],
            tasks=[
                Task("t1", "Write design doc", remaining_minutes=120, priority=3),
                Task("t2", "Answer email backlog", remaining_minutes=45, priority=1),
            ],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Optional coffee chat",
                    h2m(14, 0),
                    h2m(14, 30),
                    "Office",
                    importance=1,
                    from_person="Colleague",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Thursday",
            tags=["hard"],
        ),
        Scenario(
            scenario_id="s4_parent_pickup",
            name="School pickup constraint",
            persona_id="busy_parent",
            calendar=[
                Event("e1", "Work block", h2m(9, 0), h2m(12, 0), "Office", importance=2, kind="focus"),
                Event("e2", "School pickup", h2m(15, 0), h2m(15, 30), "School", importance=3, kind="obligation"),
            ],
            tasks=[Task("t1", "Prepare report", remaining_minutes=75, priority=3)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Last-minute meeting",
                    h2m(15, 10),
                    h2m(15, 40),
                    "Office",
                    importance=3,
                    from_person="Manager",
                    flexible=False,
                )
            ],
            travel_times=tt,
            day_of_week="Friday",
            tags=["hard"],
        ),
        Scenario(
            scenario_id="s5_late_meeting",
            name="Late meeting pressure",
            persona_id="early_bird_engineer",
            calendar=[
                Event("e1", "Sprint planning", h2m(16, 0), h2m(17, 0), "Office", importance=2),
            ],
            tasks=[Task("t1", "Refactor module", remaining_minutes=60, priority=2)],
            incoming_requests=[
                IncomingRequest(
                    "r1",
                    "Evening vendor call",
                    h2m(18, 30),
                    h2m(19, 0),
                    "Home",
                    importance=2,
                    from_person="Vendor",
                    flexible=True,
                )
            ],
            travel_times=tt,
            day_of_week="Monday",
            tags=["hard"],
        ),
    ]


def get_scenario(scenario_id: str) -> Scenario:
    for s in sample_scenarios():
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Unknown scenario_id: {scenario_id}")


def list_scenario_ids() -> List[str]:
    return [s.scenario_id for s in sample_scenarios()]


def scenario_ids_by_difficulty(difficulty: str) -> List[str]:
    """
    Curriculum helper.

    Difficulty buckets:
    - easy: 2-3 events, no travel conflicts, no goals
    - medium: 4-5 events, 1 travel constraint, 1 goal
    - hard: everything else (full constraints)
    """

    d = difficulty.strip().lower()
    all_ids = list_scenario_ids()
    if d == "easy":
        return [sid for sid in all_ids if sid.startswith("easy_")]
    if d == "medium":
        return [sid for sid in all_ids if sid.startswith("medium_")]
    if d == "hard":
        return [sid for sid in all_ids if not (sid.startswith("easy_") or sid.startswith("medium_"))]
    # Unknown difficulty => all scenarios.
    return all_ids


def edge_case_scenario_ids() -> List[str]:
    """Scenario ids for edge cases where an LLM should beat fixed rules."""
    return [sid for sid in list_scenario_ids() if sid.startswith("edge_")]

