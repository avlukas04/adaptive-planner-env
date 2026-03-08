#!/usr/bin/env python3
"""
Gradio demo for LifeOps scheduling simulation.

Pick a scenario and agent, run the simulation, and see how the agent responds.
"""

from __future__ import annotations

# Load .env before HuggingFace imports
import os as _os
_env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
if _os.path.isfile(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.lifeops_env import LifeOpsEnv, _choose_simple_action
from env.scenario_generator import get_scenario, list_scenario_ids

try:
    from agent.llm_agent import choose_action as llm_choose_action
except ImportError:
    llm_choose_action = None

try:
    from training.train_rl import collect_trajectory
except ImportError:
    collect_trajectory = None


def _min_to_time(m: int) -> str:
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def _format_calendar(calendar: List[Dict[str, Any]]) -> str:
    if not calendar:
        return "*(empty)*"
    lines = []
    for e in sorted(calendar, key=lambda x: int(x.get("start_min", 0))):
        start = int(e.get("start_min", 0))
        end = int(e.get("end_min", 0))
        lines.append(f"• {_min_to_time(start)}–{_min_to_time(end)}: {e.get('title', '?')} @ {e.get('location', '?')}")
    return "\n".join(lines)


def _format_tasks(tasks: List[Dict[str, Any]]) -> str:
    if not tasks:
        return "*(none)*"
    lines = []
    for t in tasks:
        rem = int(t.get("remaining_minutes", 0))
        if rem > 0:
            lines.append(f"• {t.get('title', '?')}: {rem} min (priority {t.get('priority', 2)})")
    return "\n".join(lines) if lines else "*(all done)*"


def _format_request(req: Optional[Dict[str, Any]]) -> str:
    if not req:
        return "*(no pending request)*"
    start = int(req.get("start_min", 0))
    end = int(req.get("end_min", 0))
    return (
        f"**{req.get('title', '?')}**\n"
        f"Time: {_min_to_time(start)}–{_min_to_time(end)} @ {req.get('location', '?')}\n"
        f"From: {req.get('from_person', '?')} | Importance: {req.get('importance', 1)}"
    )


def _format_action(a: Dict[str, Any]) -> str:
    at = a.get("action_type", "?")
    if at == "block_focus_time":
        start = a.get("new_start_min", 0)
        dur = a.get("duration_min", 0)
        h, m = divmod(start or 0, 60)
        return f"Block focus @ {h:02d}:{m:02d} ({dur} min)"
    if at == "accept_event":
        return f"Accept request {a.get('request_id', '?')}"
    if at == "reject_event":
        return f"Reject request {a.get('request_id', '?')}"
    if at == "reschedule_event":
        ns = a.get("new_start_min", 0)
        h, m = divmod(ns or 0, 60)
        return f"Reschedule → {h:02d}:{m:02d}"
    if at == "propose_new_time":
        ns = a.get("new_start_min", 0)
        h, m = divmod(ns or 0, 60)
        return f"Propose new time → {h:02d}:{m:02d}"
    return at


def _run_simulation(scenario_id: str, agent: str) -> Tuple[str, str]:
    """
    Run one episode and return (context_md, result_md).
    """
    if collect_trajectory is None:
        return "Error: Could not import training module.", ""

    scenario = get_scenario(scenario_id)
    env = LifeOpsEnv(seed=42)
    trajectory, total_reward, ep_len, _, persona_name = collect_trajectory(
        env, agent=agent, scenario_id=scenario_id
    )

    # Build context (initial state)
    obs = trajectory[0]["obs"] if trajectory else {}
    context_lines = [
        f"## Scenario: {scenario.name}",
        f"**Persona:** {persona_name}",
        "",
        "### Current calendar",
        _format_calendar(obs.get("calendar", [])),
        "",
        "### Tasks",
        _format_tasks(obs.get("tasks", [])),
        "",
        "### Incoming request",
        _format_request(obs.get("current_request")),
    ]
    context_md = "\n".join(context_lines)

    # Build result (agent actions + outcome)
    result_lines = [
        f"## Agent: {agent}",
        f"**Total reward:** {total_reward:+.2f} | **Steps:** {ep_len}",
        "",
        "### Action sequence",
    ]
    for i, t in enumerate(trajectory):
        action = t["action"]
        reward = t["reward"]
        result_lines.append(f"{i + 1}. {_format_action(action)} → reward {reward:+.2f}")

    # Final calendar
    final_obs = trajectory[-1]["next_obs"] if trajectory else {}
    result_lines.extend([
        "",
        "### Final calendar",
        _format_calendar(final_obs.get("calendar", [])),
    ])
    result_md = "\n".join(result_lines)

    return context_md, result_md


def _get_scenario_choices() -> List[Tuple[str, str]]:
    """Return [(value, label), ...] for Gradio dropdown."""
    choices = []
    for sid in list_scenario_ids():
        s = get_scenario(sid)
        choices.append((sid, f"{s.name} ({sid})"))
    return choices


def _get_agent_choices() -> List[str]:
    agents = ["random", "baseline"]
    if llm_choose_action is not None:
        agents.append("llm")
    return agents


def create_demo():
    import gradio as gr
    scenario_choices = _get_scenario_choices()
    agent_choices = _get_agent_choices()

    with gr.Blocks(title="LifeOps Scheduling Simulator") as demo:
        gr.Markdown("# LifeOps Scheduling Simulator")
        gr.Markdown(
            "Pick a scenario and agent. The agent will handle the incoming request "
            "and manage the schedule. See how different agents (random, baseline, LLM) respond."
        )

        with gr.Row():
            # Gradio Dropdown: (label, value) - label shown, value passed to fn
            scenario_gr = [(label, sid) for sid, label in scenario_choices]
            scenario_dd = gr.Dropdown(
                choices=scenario_gr,
                value=scenario_choices[0][0] if scenario_choices else None,
                label="Scenario",
            )
            agent_dd = gr.Dropdown(
                choices=agent_choices,
                value=agent_choices[0],
                label="Agent",
            )
            run_btn = gr.Button("Run simulation", variant="primary")

        with gr.Row():
            context_out = gr.Markdown(
                label="Scenario context",
                value="*Select a scenario and click Run.*",
            )
            result_out = gr.Markdown(
                label="Agent response",
                value="*Results will appear here.*",
            )

        def run(scenario_id: str, agent: str):
            if not scenario_id:
                return "Please select a scenario.", "*No scenario selected.*"
            context, result = _run_simulation(scenario_id, agent)
            return context, result

        run_btn.click(
            fn=run,
            inputs=[scenario_dd, agent_dd],
            outputs=[context_out, result_out],
        )

    return demo


if __name__ == "__main__":
    try:
        import gradio as gr
    except ImportError:
        print("Install gradio: pip install gradio")
        sys.exit(1)

    demo = create_demo()
    demo.launch(theme=gr.themes.Soft())
