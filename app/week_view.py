"""
LifeOps Week View UI.

A polished Gradio interface with:
- Week view (7 days) displaying events
- Tasks to resolve
- Persona selector
- Structure for Google Calendar integration
"""

from __future__ import annotations

import os as _os
_env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
if _os.path.isfile(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.lifeops_env import LifeOpsEnv, _choose_simple_action
from env.personas import get_personas
from env.scenario_generator import get_scenario, list_scenario_ids

try:
    from app.gcal_client import get_week_events, is_gcal_available
except ImportError:
    get_week_events = None
    is_gcal_available = lambda: False

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


def _render_week_html(
    calendar: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    pending_request: Optional[Dict[str, Any]],
    persona_name: str,
    week_start: Optional[datetime] = None,
    gcal_events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render a week view as HTML. Events are shown in 'today' column (day 3)."""
    if week_start is None:
        week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        # Start on Monday
        week_start -= timedelta(days=week_start.weekday())

    days = [(week_start + timedelta(days=i)).strftime("%a %d") for i in range(7)]
    today_idx = (datetime.now().date() - week_start.date()).days
    if today_idx < 0 or today_idx > 6:
        today_idx = 3

    # Merge GCal events into today if provided (for future sync)
    if gcal_events:
        calendar = list(calendar) + gcal_events

    # Build event blocks for "today" (the day with scenario data)
    event_html = ""
    for e in sorted(calendar, key=lambda x: int(x.get("start_min", 0))):
        start = int(e.get("start_min", 0))
        end = int(e.get("end_min", 0))
        top = (start / 1440) * 100
        height = max(4, ((end - start) / 1440) * 100)
        kind = e.get("kind", "meeting")
        color = "#6366f1" if kind == "focus" else "#0ea5e9"
        event_html += f"""
        <div class="event-block" style="top:{top}%;height:{height}%;background:{color};">
            <span class="event-time">{_min_to_time(start)}–{_min_to_time(end)}</span>
            <span class="event-title">{e.get('title', '?')}</span>
            <span class="event-loc">{e.get('location', '')}</span>
        </div>
        """

    # Pending request as a highlighted block
    request_html = ""
    if pending_request:
        start = int(pending_request.get("start_min", 0))
        end = int(pending_request.get("end_min", 0))
        top = (start / 1440) * 100
        height = max(6, ((end - start) / 1440) * 100)
        imp = int(pending_request.get("importance", 1))
        request_html = f"""
        <div class="event-block pending" style="top:{top}%;height:{height}%;background:#f59e0b;border:2px dashed #d97706;">
            <span class="event-time">{_min_to_time(start)}–{_min_to_time(end)}</span>
            <span class="event-title">📩 {pending_request.get('title', '?')}</span>
            <span class="event-loc">Importance: {imp}</span>
        </div>
        """

    # Tasks
    task_items = ""
    for t in tasks:
        rem = int(t.get("remaining_minutes", 0))
        if rem > 0:
            prio = int(t.get("priority", 2))
            task_items += f'<li><span class="task-prio p{prio}">P{prio}</span> {t.get("title", "?")} — {rem} min</li>'

    day_cells = ""
    for i, day in enumerate(days):
        is_today = "today" if i == today_idx else ""
        content = (event_html + request_html) if i == today_idx else ""
        day_cells += f"""
        <div class="day-col {is_today}">
            <div class="day-header">{day}</div>
            <div class="day-body">
                {content}
            </div>
        </div>
        """

    return f"""
    <style>
        .lifeops-week {{ font-family: 'Inter', system-ui, sans-serif; --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --accent: #6366f1; --muted: #94a3b8; }}
        .lifeops-week * {{ box-sizing: border-box; }}
        .week-container {{ display: flex; gap: 8px; margin-bottom: 24px; }}
        .day-col {{ flex: 1; background: var(--card); border-radius: 12px; overflow: hidden; min-height: 320px; }}
        .day-col.today {{ border: 2px solid var(--accent); box-shadow: 0 0 20px rgba(99,102,241,0.2); }}
        .day-header {{ padding: 12px; background: rgba(0,0,0,0.2); font-weight: 600; font-size: 13px; color: var(--text); text-align: center; }}
        .day-body {{ position: relative; height: 280px; padding: 8px; }}
        .event-block {{ position: absolute; left: 4px; right: 4px; border-radius: 8px; padding: 6px; color: white; font-size: 11px; overflow: hidden; }}
        .event-block.pending {{ animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.85; }} }}
        .event-time {{ display: block; font-weight: 600; margin-bottom: 2px; }}
        .event-title {{ display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .event-loc {{ display: block; font-size: 10px; opacity: 0.9; }}
        .tasks-panel {{ background: var(--card); border-radius: 12px; padding: 16px; margin-top: 16px; }}
        .tasks-panel h3 {{ margin: 0 0 12px; font-size: 14px; color: var(--accent); }}
        .tasks-panel ul {{ margin: 0; padding-left: 20px; color: var(--text); }}
        .task-prio {{ display: inline-block; width: 24px; text-align: center; border-radius: 4px; font-size: 10px; font-weight: 600; margin-right: 8px; }}
        .task-prio.p1 {{ background: #334155; }} .task-prio.p2 {{ background: #475569; }} .task-prio.p3 {{ background: var(--accent); }}
        .persona-badge {{ display: inline-block; padding: 6px 12px; background: var(--accent); color: white; border-radius: 8px; font-size: 13px; font-weight: 500; }}
    </style>
    <div class="lifeops-week">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <h2 style="margin:0;color:var(--text);font-size:20px;">LifeOps Schedule</h2>
            <span class="persona-badge">{persona_name}</span>
        </div>
        <div class="week-container">
            {day_cells}
        </div>
        <div class="tasks-panel">
            <h3>📋 Tasks to resolve</h3>
            <ul>{task_items if task_items else "<li style='color:var(--muted);'>No pending tasks</li>"}</ul>
        </div>
    </div>
    """


def _run_and_get_state(scenario_id: str, agent: str) -> Dict[str, Any]:
    """Run simulation and return final state for display."""
    if collect_trajectory is None:
        return {}
    env = LifeOpsEnv(seed=42)
    trajectory, total_reward, ep_len, _, persona_name = collect_trajectory(
        env, agent=agent, scenario_id=scenario_id
    )
    if not trajectory:
        return {}
    final = trajectory[-1]["next_obs"]
    return {
        "calendar": final.get("calendar", []),
        "tasks": final.get("tasks", []),
        "persona_name": persona_name,
        "total_reward": total_reward,
        "ep_len": ep_len,
    }


def _get_initial_state(scenario_id: str) -> Dict[str, Any]:
    """Get initial state from scenario (before agent runs)."""
    scenario = get_scenario(scenario_id)
    calendar = [e.to_dict() for e in scenario.calendar]
    tasks = [t.to_dict() for t in scenario.tasks]
    pending = scenario.incoming_requests
    req = pending[0].to_dict() if pending else None
    personas = get_personas()
    persona = personas.get(scenario.persona_id)
    return {
        "calendar": calendar,
        "tasks": tasks,
        "current_request": req,
        "persona_name": persona.name if persona else "?",
    }


def create_week_demo():
    import gradio as gr

    scenario_choices = [(sid, f"{get_scenario(sid).name} ({sid})") for sid in list_scenario_ids()]
    agents = ["random", "baseline"] + (["llm"] if llm_choose_action else [])

    with gr.Blocks(
        title="LifeOps",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # LifeOps — AI Schedule Assistant
            Manage your week, resolve conflicts, and block focus time.
            """
        )

        gcal_status = "🔗 Connect GCal" if (get_week_events and is_gcal_available()) else "📅 GCal: add credentials.json to enable"
        gr.Markdown(f"*{gcal_status}*")

        personas = get_personas()
        persona_choices = [(p.name, p.persona_id) for p in personas.values()]

        with gr.Row():
            persona_dd = gr.Dropdown(
                choices=[(n, pid) for n, pid in persona_choices],
                value=persona_choices[0][1] if persona_choices else None,
                label="Persona",
            )
            scenario_dd = gr.Dropdown(
                choices=[(l, v) for v, l in scenario_choices],
                value=scenario_choices[0][0] if scenario_choices else None,
                label="Scenario",
            )
            agent_dd = gr.Dropdown(choices=agents, value=agents[0], label="Agent")
            run_btn = gr.Button("Run AI", variant="primary")

        with gr.Tabs():
            with gr.TabItem("Week view"):
                week_html = gr.HTML(
                    value=_render_week_html([], [], None, "Select a scenario"),
                    label="Schedule",
                )
            with gr.TabItem("Results"):
                result_md = gr.Markdown(value="*Run the simulation to see results.*")

        def _filter_scenarios_by_persona(persona_id: Optional[str]) -> list:
            if not persona_id:
                return [(l, v) for v, l in scenario_choices]
            return [(l, v) for v, l in scenario_choices if get_scenario(v).persona_id == persona_id]

        def on_persona_change(persona_id: str):
            filtered = _filter_scenarios_by_persona(persona_id)
            return gr.Dropdown(choices=filtered, value=filtered[0][0] if filtered else None)

        def on_scenario_change(scenario_id: str):
            if not scenario_id:
                return _render_week_html([], [], None, "—")
            state = _get_initial_state(scenario_id)
            return _render_week_html(
                state["calendar"],
                state["tasks"],
                state["current_request"],
                state["persona_name"],
            )

        def on_run(scenario_id: str, agent: str):
            if not scenario_id:
                return _render_week_html([], [], None, "—"), "*Select a scenario.*"
            state = _run_and_get_state(scenario_id, agent)
            if not state:
                return on_scenario_change(scenario_id), "*Error running simulation.*"
            html = _render_week_html(
                state["calendar"],
                state["tasks"],
                None,
                state["persona_name"],
            )
            result = f"**Reward:** {state['total_reward']:+.2f} | **Steps:** {state['ep_len']}"
            return html, result

        persona_dd.change(
            fn=on_persona_change,
            inputs=[persona_dd],
            outputs=[scenario_dd],
        )

        scenario_dd.change(
            fn=on_scenario_change,
            inputs=[scenario_dd],
            outputs=[week_html],
        )

        run_btn.click(
            fn=on_run,
            inputs=[scenario_dd, agent_dd],
            outputs=[week_html, result_md],
        )

        # Initial load
        demo.load(
            fn=on_scenario_change,
            inputs=[scenario_dd],
            outputs=[week_html],
        )

    return demo


if __name__ == "__main__":
    demo = create_week_demo()
    demo.launch()
