"""
LifeOps Streamlit demo UI.

Run:
  pip install -r requirements.txt
  streamlit run demo.py

Design goals:
- Hackathon-friendly: simple, readable, no heavy frontend.
- Light shadcn/ui-like aesthetic (white cards, rounded corners, subtle shadows).
- Uses env/ and llm_agent.py directly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from env.actions import mask_illegal_actions
from env.baseline_agent import choose_baseline_action
from env.episode_trace import EpisodeTrace
from env.lifeops_env import LifeOpsEnv
from env.scenario_generator import list_scenario_ids
from env.reward import detect_overlaps, travel_issues
from llm_agent import LLMAgent


def _min_to_hhmm(m: int) -> str:
    h, mm = divmod(int(m), 60)
    h12 = h % 12
    h12 = 12 if h12 == 0 else h12
    ampm = "am" if h < 12 else "pm"
    return f"{h12}:{mm:02d}{ampm}"


def _event_color(e: Dict[str, Any]) -> str:
    """
    Color mapping (as requested):
    - work meetings = blue
    - personal = green
    - goals = purple
    """

    kind = str(e.get("kind", "meeting"))
    if kind == "personal":
        return "#2ecc71"  # green
    if kind == "focus":
        return "#a855f7"  # purple
    # meeting / obligation / default
    return "#60a5fa"  # blue


def _compute_conflict_event_ids(state: Dict[str, Any], calendar: List[Dict[str, Any]]) -> Tuple[set, Dict[str, int]]:
    """
    Compute constraint violations for highlighting in the BEFORE view.

    Conflicts include:
    - overlapping events (double booking)
    - meeting-like events scheduled during low-productivity hours (before 9am, 8pm-11pm)
    - travel infeasibility between consecutive events
    """

    blocked = ((0, 9 * 60), (20 * 60, 23 * 60))

    def overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
        return a_s < b_e and b_s < a_e

    ids = set()
    counts = {"overlap": 0, "blocked_hours": 0, "travel": 0}

    # Overlaps
    pairs = detect_overlaps(calendar)
    counts["overlap"] = len(pairs)
    for a, b in pairs:
        ids.add(a)
        ids.add(b)

    # Blocked meeting hours
    for e in calendar:
        kind = str(e.get("kind", "meeting"))
        if kind not in {"meeting", "obligation", "personal"}:
            continue
        s = int(e.get("start_min", 0))
        en = int(e.get("end_min", 0))
        for bs, be in blocked:
            if overlaps(s, en, bs, be):
                ids.add(str(e.get("event_id", "")))
                counts["blocked_hours"] += 1
                break

    # Travel infeasibility
    travel = state.get("travel_times", {}) or {}
    home = (state.get("persona", {}) or {}).get("home_location")
    issues = travel_issues(calendar, travel, start_location=home)
    counts["travel"] = len(issues)
    for a, b, *_ in issues:
        ids.add(a)
        ids.add(b)

    return ids, counts


def render_timeline_html(
    state: Dict[str, Any],
    calendar: List[Dict[str, Any]],
    *,
    issue_levels: Optional[Dict[str, str]] = None,
    moved_event_ids: Optional[set] = None,
    day_start: int = 8 * 60,
    day_end: int = 23 * 60,
) -> str:
    """
    Shadcn-like daily timeline: each hour is a row, events are full-width pill rows.

    Events are rendered as horizontal rows:
      [3px colored left border] Event Name .......... 9:00am–10:00am
    This prevents overflow and truncates cleanly on narrow widths.
    """

    def overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
        return a_s < b_e and b_s < a_e

    events = sorted(calendar, key=lambda e: (int(e.get("start_min", 0)), int(e.get("end_min", 0))))
    hours = list(range(day_start, day_end + 1, 60))

    def row(e: Dict[str, Any], level: str, moved: bool) -> str:
        title = str(e.get("title", e.get("event_id", "Event")))
        s = int(e.get("start_min", 0))
        en = int(e.get("end_min", 0))
        time = f"{_min_to_hhmm(s)}–{_min_to_hhmm(en)}"
        accent = _event_color(e)
        if level == "conflict":
            accent = "#ef4444"
        elif level == "lowprod":
            accent = "#f59e0b"

        moved_txt = " · ↗ moved" if moved else ""

        # Requirement: simple inline styles only (no classes) for event bubbles.
        # Also ensure no overflow outside the calendar column.
        bg = f"{accent}15"
        text = f"{title} · {time}{moved_txt}"
        return (
            "<div style=\""
            f"border-left: 3px solid {accent};"
            f"background: {bg};"
            "padding: 4px 8px;"
            "border-radius: 4px;"
            "margin: 2px 0;"
            "font-size: 13px;"
            "overflow: hidden;"
            "text-overflow: ellipsis;"
            "white-space: nowrap;"
            "width: 100%;"
            "box-sizing: border-box;"
            "\">"
            + text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            + "</div>"
        )

    rows = []
    for h in hours:
        h_end = h + 60
        label = _min_to_hhmm(h)
        row_events = []
        for e in events:
            s = int(e.get("start_min", 0))
            en = int(e.get("end_min", 0))
            if overlaps(s, en, h, h_end):
                eid = str(e.get("event_id", ""))
                level = (issue_levels or {}).get(eid, "normal")
                moved = moved_event_ids is not None and eid in moved_event_ids
                row_events.append(row(e, level=level, moved=moved))
        # Clamp visual density
        extra = ""
        if len(row_events) > 3:
            extra = f"<span class='more'>+{len(row_events) - 3} more</span>"
            row_events = row_events[:3]
        # Inline styles for robust layout + clipping (no overflow).
        rows.append(
            "<div style=\"display:flex;gap:12px;align-items:flex-start;"
            "border-top:1px solid #e5e7eb;padding:6px 0;\">"
            f"<div style=\"width:84px;flex:0 0 auto;font-size:12px;color:#64748b;\">{label}</div>"
            "<div style=\"flex:1 1 auto;min-width:0;overflow:hidden;\">"
            + "".join(row_events)
            + extra
            + "</div></div>"
        )

    day = str(state.get("day_of_week", ""))
    day_line = f"<div style='font-size:12px;color:#64748b;margin-bottom:6px;'>Day: {day}</div>" if day else ""
    return "<div>" + day_line + "".join(rows) + "</div>"


def _issue_levels_for_before(state: Dict[str, Any], calendar: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Return per-event issue level for BEFORE view:
    - conflict: overlaps or travel infeasible
    - lowprod: meeting-like event in blocked hours (only if not already conflict)
    """

    ids, counts = _compute_conflict_event_ids(state, calendar)
    levels: Dict[str, str] = {str(eid): "conflict" for eid in ids}

    blocked = ((0, 9 * 60), (20 * 60, 23 * 60))

    def overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
        return a_s < b_e and b_s < a_e

    for e in calendar:
        eid = str(e.get("event_id", ""))
        if levels.get(eid) == "conflict":
            continue
        kind = str(e.get("kind", "meeting"))
        if kind not in {"meeting", "obligation", "personal"}:
            continue
        s = int(e.get("start_min", 0))
        en = int(e.get("end_min", 0))
        for bs, be in blocked:
            if overlaps(s, en, bs, be):
                levels[eid] = "lowprod"
                break

    return levels


def _moved_event_ids_from_trajectory(trajectory: List[Dict[str, Any]]) -> set:
    moved = set()
    for step in trajectory:
        a = step.get("action") or {}
        if str(a.get("action_type")) == "reschedule_event":
            rid = a.get("request_id")
            if rid:
                moved.add(str(rid))
    return moved


def _what_fixed_cards_html(agent: LLMAgent, trajectory: List[Dict[str, Any]]) -> str:
    """
    Build 2-3 small cards describing what the agent fixed.
    """

    if not trajectory:
        return "<div style='color:#64748b;font-size:13px'>Run the agent to see conflicts resolved</div>"

    def desc(step: Dict[str, Any]) -> Tuple[str, str, float]:
        obs = step.get("obs") or {}
        a = step.get("action") or {}
        info = step.get("info") or {}
        rb = info.get("reward_breakdown") or {}
        req = obs.get("current_request") or {}
        title = str(req.get("title", "request"))
        t = _min_to_hhmm(int(req.get("start_min", a.get("new_start_min") or 0)))
        r = float(step.get("reward", 0.0))
        at = str(a.get("action_type", ""))

        # reuse a simplified reason from action history logic
        day = str(obs.get("day_of_week", ""))
        bucket = (int(req.get("start_min", 0)) // 60) * 60
        row = agent.preference_model.get((day, bucket), {}) if getattr(agent, "preference_model", None) else {}
        accepted = int(row.get("accepted", 0))
        rejected = int(row.get("rejected", 0))

        if at == "reject_event":
            reason = "Avoid conflict / low value"
            if int(req.get("start_min", 0)) < 10 * 60 and (accepted + rejected) >= 2 and rejected >= accepted:
                reason = "user cancels before 10am ~70% of the time"
            icon = "🔴"
            return (icon, f"{t} {title} removed — {reason}", r)
        if at in {"reschedule_event", "propose_new_time"}:
            icon = "🟠"
            return (icon, f"{t} {title} moved ↗ — avoided conflict/travel", r)
        if at == "block_focus_time":
            icon = "🟠"
            if rb.get("deadline_pressure_penalty"):
                return (icon, f"Goal time protected — urgent deadline", r)
            return (icon, f"Goal block protected — improved productivity", r)
        # fallback
        return ("🟠", f"{t} {title} improved schedule", r)

    candidates = [step for step in trajectory if float(step.get("reward", 0.0)) > 0.0]
    candidates.sort(key=lambda s: float(s.get("reward", 0.0)), reverse=True)
    top = candidates[:3]
    if not top:
        return "<div style='color:#64748b;font-size:13px'>Run the agent to see conflicts resolved</div>"

    cards = []
    for s in top:
        icon, text, r = desc(s)
        cards.append(
            f"""
<div class="fixcard">
  <div class="fixicon">{icon}</div>
  <div class="fixtext">{text}</div>
  <div class="fixreward">{r:+.0f} pts</div>
</div>
"""
        )

    return f"""
<style>
  .fixwrap {{ display:flex; flex-direction:column; gap:10px; margin-top:10px; }}
  .fixcard {{
    display:flex; align-items:center; gap:12px;
    background:#f1f5f9; border-radius:16px; padding:12px 12px;
  }}
  .fixicon {{ width:24px; text-align:center; }}
  .fixtext {{ flex:1; font-size:13px; color:#0f172a; }}
  .fixreward {{ font-weight:800; color:#16a34a; font-size:13px; }}
  .badge-mini {{ padding: 3px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge-red {{ background: rgba(239,68,68,0.12); color:#991b1b; }}
  .badge-orange {{ background: rgba(245,158,11,0.16); color:#92400e; }}
  .moved {{ font-size:11px; color:#64748b; font-weight:700; margin-left:6px; }}
</style>
<div class="fixwrap">{''.join(cards)}</div>
"""


def _format_decision_line(agent: LLMAgent, step: Dict[str, Any]) -> str:
    """
    Format like:
      "8am meeting → REJECTED — user cancels these 70% of the time"
    """

    obs = step.get("obs") or {}
    action = step.get("action") or {}
    req = obs.get("current_request") or {}
    title = str(req.get("title", "request"))
    start = _min_to_hhmm(int(req.get("start_min", 0)))
    at = str(action.get("action_type", "")).upper()

    # Preference % (day+hour bucket)
    day = str(obs.get("day_of_week", ""))
    bucket = (int(req.get("start_min", 0)) // 60) * 60
    row = agent.preference_model.get((day, bucket), {}) if getattr(agent, "preference_model", None) else {}
    accepted = int(row.get("accepted", 0))
    rejected = int(row.get("rejected", 0))
    total = accepted + rejected
    if total >= 1:
        cancel_pct = int(round(100.0 * (rejected / float(total))))
        reason = f"user cancels these ~{cancel_pct}% of the time"
    else:
        reason = "no prior preference data yet"

    arrow = "→"
    return f"{start} {title} {arrow} {at} — {reason}"


def _preferences_bullets(agent: LLMAgent, day: str) -> List[str]:
    if not getattr(agent, "preference_model", None):
        return ["No learned preferences yet."]

    scored: List[Tuple[int, float, int]] = []
    for (d, b), row in agent.preference_model.items():
        if d != day:
            continue
        seen = int(row.get("accepted", 0) + row.get("rejected", 0))
        if seen < 2:
            continue
        score = float(row.get("accepted", 0) - row.get("rejected", 0)) + 0.5 * float(row.get("good", 0) - row.get("bad", 0))
        scored.append((int(b), score, seen))

    if not scored:
        return [f"No stable preferences learned for {day} yet."]

    # IMPORTANT: avoid contradictory output.
    # If there are only a few buckets, a naive "top 3" and "bottom 3" can overlap
    # (e.g., the same time listed as both Prefer and Avoid). We enforce disjoint
    # sets and bias toward clearly positive vs clearly negative scores.
    scored.sort(key=lambda t: t[1], reverse=True)

    positive = [t for t in scored if t[1] > 0.25]
    negative = [t for t in sorted(scored, key=lambda t: t[1]) if t[1] < -0.25]

    top = positive[:3]
    top_set = {b for b, _, _ in top}
    bottom = [t for t in negative if t[0] not in top_set][:3]

    def fmt(items: List[Tuple[int, float, int]]) -> str:
        return ", ".join(f"{_min_to_hhmm(b)} (score {s:+.1f}, {seen}x)" for b, s, seen in items)

    lines: List[str] = []
    lines.append(f"Prefer: {fmt(top)}" if top else "Prefer: (none yet)")
    lines.append(f"Avoid: {fmt(bottom)}" if bottom else "Avoid: (none yet)")
    return lines


def _episode_summary_line(before_obs: Dict[str, Any], after_obs: Dict[str, Any]) -> str:
    """
    Short plain-English before/after summary shown between views.
    """

    before_ids, before_counts = _compute_conflict_event_ids(before_obs, list(before_obs.get("calendar", [])))
    after_ids, after_counts = _compute_conflict_event_ids(after_obs, list(after_obs.get("calendar", [])))

    resolved = max(0, int(before_counts.get("overlap", 0)) - int(after_counts.get("overlap", 0)))
    avoided_travel = max(0, int(before_counts.get("travel", 0)) - int(after_counts.get("travel", 0)))

    def ok_focus_blocks(obs: Dict[str, Any]) -> int:
        cal = list(obs.get("calendar", []))
        conflict_ids, _ = _compute_conflict_event_ids(obs, cal)
        focus = [e for e in cal if str(e.get("kind", "")) == "focus"]
        return sum(1 for e in focus if str(e.get("event_id", "")) not in conflict_ids)

    protected = max(0, ok_focus_blocks(after_obs) - ok_focus_blocks(before_obs))
    return f"Agent resolved {resolved} conflicts, protected {protected} goal block, avoided {avoided_travel} bad travel window"


def _clipboard_export_button(label: str, text: str) -> None:
    """
    Render a small button that copies `text` to clipboard.
    """
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    components.html(
        f"""
<div style="margin-top: 6px; margin-bottom: 6px;">
  <button id="copyBtn" style="
    background:#18181b;color:#fafafa;border:0;
    padding:10px 12px;border-radius:999px;cursor:pointer;
    font-weight:600;">
    {label}
  </button>
  <span id="copyMsg" style="margin-left:10px;color:#94a3b8;font-size:13px;"></span>
</div>
<script>
  const btn = document.getElementById("copyBtn");
  const msg = document.getElementById("copyMsg");
  const payload = `{escaped}`;
  btn.onclick = async () => {{
    try {{
      await navigator.clipboard.writeText(payload);
      msg.textContent = "Copied to clipboard.";
      setTimeout(() => msg.textContent = "", 1500);
    }} catch (e) {{
      msg.textContent = "Copy failed (browser blocked clipboard).";
    }}
  }};
</script>
""",
        height=50,
    )


def _action_history_table_html(agent: LLMAgent, trajectory: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    """
    Return (html_table, tsv_export, total_reward).
    """

    def reason_for(step: Dict[str, Any]) -> str:
        obs = step.get("obs") or {}
        action = step.get("action") or {}
        info = step.get("info") or {}
        rb = info.get("reward_breakdown") or {}
        req = obs.get("current_request") or {}
        start = int(req.get("start_min", 0))
        day = str(obs.get("day_of_week", ""))
        bucket = (start // 60) * 60
        row = agent.preference_model.get((day, bucket), {}) if getattr(agent, "preference_model", None) else {}
        accepted = int(row.get("accepted", 0))
        rejected = int(row.get("rejected", 0))

        at = str(action.get("action_type", ""))
        if at == "reject_event":
            if start < 10 * 60 and (rejected >= accepted) and (accepted + rejected) >= 2:
                return "User cancels before 10am"
            if rb.get("cascade_penalty"):
                return "Avoid cascading conflict"
            return "Avoid conflict / low value"
        if at == "accept_event":
            if rb.get("feasible_schedule_bonus"):
                return "No conflicts, feasible schedule"
            return "Accept when feasible"
        if at in {"reschedule_event", "propose_new_time"}:
            if rb.get("inflexible_reschedule_penalty"):
                return "Protect inflexible obligations"
            return "Move to avoid conflicts/travel"
        if at == "block_focus_time":
            if rb.get("deadline_pressure_penalty"):
                return "Urgent goal deadline"
            return "Protect goal block"
        return "Best available option"

    total_reward = sum(float(s.get("reward", 0.0)) for s in trajectory)

    rows_html: List[str] = []
    tsv_lines = ["Time\tEvent\tAction\tReason\tReward"]
    for s in trajectory:
        obs = s.get("obs") or {}
        action = s.get("action") or {}
        req = obs.get("current_request") or {}
        at = str(action.get("action_type", ""))
        time_str = _min_to_hhmm(int(req.get("start_min", action.get("new_start_min") or 0)))
        event = str(req.get("title", "Focus")) if at != "block_focus_time" else "Focus block"
        reward = float(s.get("reward", 0.0))
        reason = reason_for(s)

        badge = {
            "accept_event": "<span class='badge badge-green'>Accepted</span>",
            "reject_event": "<span class='badge badge-red'>Rejected</span>",
            "reschedule_event": "<span class='badge badge-yellow'>Rescheduled</span>",
            "propose_new_time": "<span class='badge badge-yellow'>Rescheduled</span>",
            "block_focus_time": "<span class='badge badge-purple'>Focus</span>",
        }.get(at, f"<span class='badge'>{at}</span>")

        # Zebra rows
        row_idx = len(rows_html)
        zebra = "row-alt" if (row_idx % 2 == 1) else ""

        rows_html.append(
            f"<tr class='{zebra}'>"
            f"<td>{time_str}</td>"
            f"<td>{event}</td>"
            f"<td>{badge}</td>"
            f"<td>{reason}</td>"
            f"<td style='text-align:right'>{reward:+.2f}</td>"
            f"</tr>"
        )
        tsv_lines.append(f"{time_str}\t{event}\t{at}\t{reason}\t{reward:+.2f}")

    rows_html.append(
        f"<tr class='row-total'>"
        f"<td colspan='4'><b>TOTAL REWARD</b></td>"
        f"<td style='text-align:right'><b>{total_reward:+.2f}</b></td>"
        f"</tr>"
    )
    tsv_lines.append(f"TOTAL\t\t\t\t{total_reward:+.2f}")

    table = f"""
<style>
  table.lifeops {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  table.lifeops th, table.lifeops td {{ padding: 12px 10px; border-bottom: 1px solid #e5e7eb; }}
  table.lifeops thead th {{ background: #f1f5f9; color: #0f172a; font-weight: 700; }}
  table.lifeops td {{ color: #0f172a; }}
  .row-alt td {{ background: #fafafa; }}
  .row-total td {{ background: #f1f5f9; }}
  .badge {{ display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; font-weight:600; font-size:12px; }}
  .badge-green {{ background: rgba(34,197,94,0.12); color:#166534; }}
  .badge-red {{ background: rgba(239,68,68,0.12); color:#991b1b; }}
  .badge-yellow {{ background: rgba(234,179,8,0.14); color:#854d0e; }}
  .badge-purple {{ background: rgba(168,85,247,0.12); color:#6b21a8; }}
</style>
<table class="lifeops">
  <thead>
    <tr>
      <th>Time</th>
      <th>Event</th>
      <th>Action</th>
      <th>Reason</th>
      <th style="text-align:right">Reward</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
"""
    return table, "\n".join(tsv_lines), float(total_reward)


def run_one_episode(
    env: LifeOpsEnv, agent_kind: str, llm_agent: LLMAgent
) -> Tuple[EpisodeTrace, float, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    obs = env.observation()
    before_obs = obs
    trace = EpisodeTrace(
        scenario_id=obs["scenario_id"],
        persona_name=obs["persona"]["name"],
        initial_calendar=list(obs.get("calendar", [])),
        initial_tasks=list(obs.get("tasks", [])),
        initial_pending_count=int(obs.get("pending_request_count", 0)),
    )

    total_reward = 0.0
    done = False
    step_num = 0
    rng = random.Random()
    trajectory: List[Dict[str, Any]] = []

    while not done:
        prev_obs = env.observation()
        valid = env.valid_actions()
        valid = mask_illegal_actions(prev_obs, valid)

        if agent_kind == "random":
            action = rng.choice(valid)
        elif agent_kind == "baseline":
            action = choose_baseline_action(prev_obs, valid)
        else:
            action = llm_agent.choose_action(env)

        next_obs, reward, done, info = env.step(action)
        step_num += 1
        total_reward += float(reward)

        trajectory.append(
            {
                "obs": prev_obs,
                "action": action.to_dict(),
                "reward": float(reward),
                "info": info,
            }
        )

        trace.log_step(
            step=step_num,
            action=action.to_dict(),
            prev_obs=prev_obs,
            next_obs=next_obs,
            reward=float(reward),
            breakdown=info.get("reward_breakdown", {}),
            info=info,
            done=bool(done),
            last_added_event=info.get("last_added_event"),
            last_handled_request=info.get("last_handled_request"),
            last_task_progress_minutes=int(info.get("last_task_progress_minutes", 0)),
            task_id_progressed=info.get("last_task_id_progressed"),
        )

    trace.total_reward = float(total_reward)
    after_obs = env.observation()

    # Update LLM memory (only if the episode was driven by llm agent).
    if agent_kind == "llm":
        llm_agent.on_episode_end(trajectory, total_reward=float(total_reward))

    return trace, float(total_reward), trajectory, before_obs, after_obs


def main() -> None:
    st.set_page_config(page_title="LifeOps Demo", page_icon="🗓️", layout="wide")

    # Light shadcn/ui-like styling.
    st.markdown(
        """
<style>
  :root {
    --bg: #ffffff;
    --card: #ffffff;
    --text: #18181b;    /* zinc-900 */
    --muted: #64748b;   /* slate-500 */
    --muted2: #94a3b8;  /* slate-400 */
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    --radius: 24px;
  }

  .stApp { background: var(--bg); color: var(--text); }
  .block-container { padding-top: 0.75rem; }

  /* Ensure our header is NOT sticky.
     Streamlit's built-in header bar is sticky by default; hide it so only our
     in-page header card remains (it scrolls away naturally). */
  header[data-testid="stHeader"] { display: none; }
  div[data-testid="stToolbar"] { display: none; }

  /* Typography */
  h1, h2, h3, p, label, div, span { color: var(--text); }
  .caption { color: var(--muted); }

  /* Card-like containers: use Streamlit border wrapper as a card. */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: none !important;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
  }

  /* Make the two main cards (calendar + brain) pop slightly more. */
  div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2),
  div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3) {
    box-shadow: 0 14px 40px rgba(15, 23, 42, 0.16);
  }
  div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 18px 18px 14px 18px;
  }

  /* Selectboxes as rounded "pills" */
  div[data-testid="stSelectbox"] > div {
    border-radius: 999px !important;
  }
  div[data-baseweb="select"] > div {
    border-radius: 999px !important;
    background: #f1f5f9 !important;
    border: none !important;
  }

  /* Primary button as black pill */
  button[kind="primary"] {
    background: #18181b !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.55rem 1rem !important;
  }
  button[kind="primary"] * { color: #ffffff !important; }
  button {
    border-radius: 999px !important;
  }

  /* Segmented control-like radio */
  div[role="radiogroup"] {
    background: #f1f5f9;
    border-radius: 999px;
    padding: 4px;
    display: inline-flex;
    gap: 6px;
  }
  div[role="radiogroup"] label {
    background: transparent;
    border-radius: 999px;
    padding: 6px 12px;
    margin: 0 !important;
  }

  /* Timeline styles (HTML) */
  /* Timeline uses inline styles for event rows (requested). */
</style>
""",
        unsafe_allow_html=True,
    )

    if "env" not in st.session_state:
        st.session_state.env = LifeOpsEnv(seed=42)
        st.session_state.env.reset()
    if "llm_agent" not in st.session_state:
        st.session_state.llm_agent = LLMAgent(local_model_name=None)
    if "trace" not in st.session_state:
        st.session_state.trace = None
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = []
    if "before_obs" not in st.session_state:
        st.session_state.before_obs = None
    if "after_obs" not in st.session_state:
        st.session_state.after_obs = None
    if "episode_reward" not in st.session_state:
        st.session_state.episode_reward = 0.0

    env: LifeOpsEnv = st.session_state.env
    llm_agent: LLMAgent = st.session_state.llm_agent

    # HEADER (clean white top bar)
    header = st.container(border=True)
    with header:
        c0, c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.8, 0.9, 1.0])
        with c0:
            st.markdown("<div style='font-size:22px;font-weight:800'>LifeOps</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-top:-2px;font-size:12px;color:#64748b'>AI Agent that learns your scheduling habits</div>",
                unsafe_allow_html=True,
            )
        with c1:
            agent_kind = st.selectbox("Agent", ["llm", "baseline", "random"], index=0, label_visibility="collapsed")
        with c2:
            scenario_labels = {
                "easy_1": "Simple Day (3 events)",
                "medium_1": "Busy Day (5 events)",
                "edge_cascade_1": "Conflict Day (cascading conflicts)",
                "edge_mood_monday_1": "Monday Blues (mood patterns)",
                "edge_deadline_1": "Crunch Week (goal deadline pressure)",
            }
            scenario_ids = list_scenario_ids()
            scenario = st.selectbox(
                "Scenario",
                scenario_ids,
                index=0,
                label_visibility="collapsed",
                format_func=lambda sid: scenario_labels.get(sid, sid.replace("_", " ").title()),
            )
        with c3:
            if st.button("Reset", use_container_width=True):
                env.reset(scenario_id=scenario)
                st.session_state.trace = None
                st.session_state.trajectory = []
                st.session_state.before_obs = None
                st.session_state.after_obs = None
                st.session_state.episode_reward = 0.0
        with c4:
            run_click = st.button("▶ Run Agent", type="primary", use_container_width=True)

    # Main layout
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        cal_card = st.container(border=True)
        with cal_card:
            st.subheader("Live Calendar")
            st.caption("Events update as the agent accepts or rejects them")

            if run_click:
                env.reset(scenario_id=scenario)
                trace, total, traj, before_obs, after_obs = run_one_episode(env, agent_kind=agent_kind, llm_agent=llm_agent)
                st.session_state.trace = trace
                st.session_state.trajectory = traj
                st.session_state.before_obs = before_obs
                st.session_state.after_obs = after_obs
                st.session_state.episode_reward = total

            before_obs = st.session_state.before_obs or env.observation()
            after_obs = st.session_state.after_obs or env.observation()

            before_calendar = list(before_obs.get("calendar", []))
            req = before_obs.get("current_request")
            if req is not None:
                tentative = dict(req)
                tentative["event_id"] = "__request__"
                before_calendar.append(tentative)

            # Build issue levels for BEFORE (conflict vs low productivity hours).
            before_levels = _issue_levels_for_before(before_obs, before_calendar)

            traj: List[Dict[str, Any]] = st.session_state.trajectory or []
            moved_ids = _moved_event_ids_from_trajectory(traj)

            after_calendar = list(after_obs.get("calendar", []))
            after_levels: Dict[str, str] = {}  # "After" should be clean; show conflict only if any remain
            after_conf_ids, _ = _compute_conflict_event_ids(after_obs, after_calendar)
            for eid in after_conf_ids:
                after_levels[str(eid)] = "conflict"

            before_html = render_timeline_html(before_obs, before_calendar, issue_levels=before_levels)
            after_html = render_timeline_html(after_obs, after_calendar, issue_levels=after_levels, moved_event_ids=moved_ids)

            # Two timelines side-by-side at exactly 50% / 50%, with a centered divider+arrow overlay.
            st.markdown(
                f"""
<style>
  .ba3 {{
    position: relative;
    display: flex;
    gap: 0;
    width: 100%;
    margin-top: 6px;
  }}
  .ba3-col {{
    width: 50%;
    overflow: hidden;
    padding-right: 10px;
    box-sizing: border-box;
    min-width: 0;
  }}
  .ba3-col.right {{
    padding-left: 10px;
    padding-right: 0;
  }}
  .ba3-divider {{
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 0;
    pointer-events: none;
  }}
  .ba3-divider:before {{
    content: "";
    position: absolute;
    left: -0.5px;
    top: 0;
    bottom: 0;
    width: 1px;
    background: #e5e7eb;
  }}
  .ba3-arrow {{
    position: absolute;
    left: -18px;
    top: 50%;
    transform: translateY(-50%);
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: 6px 10px;
    color: #64748b;
    font-weight: 800;
  }}
  .ba3-head {{
    font-weight: 800;
    margin-bottom: 6px;
  }}
</style>
<div class="ba3">
  <div class="ba3-col left">
    <div class="ba3-head" style="color:#64748b">⚠️ Before</div>
    {before_html}
  </div>
  <div class="ba3-col right">
    <div class="ba3-head" style="color:#16a34a">✓ After</div>
    {after_html}
  </div>
  <div class="ba3-divider"><div class="ba3-arrow">→</div></div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.caption(_episode_summary_line(before_obs, after_obs))

            st.markdown(
                "<div style='margin-top:8px;color:#64748b;font-size:12px'>"
                "<span style='display:inline-flex;align-items:center;gap:8px;margin-right:14px'>"
                "<span style='width:10px;height:10px;border-radius:999px;background:#60a5fa;display:inline-block'></span>Work</span>"
                "<span style='display:inline-flex;align-items:center;gap:8px;margin-right:14px'>"
                "<span style='width:10px;height:10px;border-radius:999px;background:#2ecc71;display:inline-block'></span>Personal</span>"
                "<span style='display:inline-flex;align-items:center;gap:8px;margin-right:14px'>"
                "<span style='width:10px;height:10px;border-radius:999px;background:#a855f7;display:inline-block'></span>Goals</span>"
                "<span style='display:inline-flex;align-items:center;gap:8px;margin-right:14px'>"
                "<span style='width:10px;height:10px;border-radius:999px;background:#ef4444;display:inline-block'></span>Conflict</span>"
                "<span style='display:inline-flex;align-items:center;gap:8px;margin-right:14px'>"
                "<span style='width:10px;height:10px;border-radius:999px;background:#f59e0b;display:inline-block'></span>Low hours</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-weight:800'>What the Agent Fixed</div>", unsafe_allow_html=True)
            st.markdown(_what_fixed_cards_html(llm_agent, traj), unsafe_allow_html=True)

    with right:
        brain_card = st.container(border=True)
        with brain_card:
            st.subheader("Agent Brain")
            st.caption("Watch the AI explain its decisions in real time")
            st.markdown(
                f"<div style='font-size:34px;font-weight:800;margin-top:2px'>{st.session_state.episode_reward:+.2f}</div>",
                unsafe_allow_html=True,
            )

            traj: List[Dict[str, Any]] = st.session_state.trajectory or []
            if not traj:
                st.write("Run an episode to see decisions and learned preferences.")
            else:
                st.markdown("**Last 5 decisions**")
                for step in traj[-5:][::-1]:
                    a = (step.get("action") or {}).get("action_type", "")
                    dot = "#22c55e" if a == "accept_event" else ("#ef4444" if a == "reject_event" else ("#eab308" if a in {"reschedule_event", "propose_new_time"} else "#a855f7"))
                    line = _format_decision_line(llm_agent, step)
                    st.markdown(
                        f"<div style='display:flex;gap:10px;align-items:center;padding:8px 0;border-top:1px solid #e5e7eb'>"
                        f"<span style='width:10px;height:10px;border-radius:999px;background:{dot};display:inline-block'></span>"
                        f"<span style='font-size:13px;color:#0f172a'>{line}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("**Learned user preferences**")
                st.caption("Patterns the agent has discovered about this user")
                day = str(env.observation().get("day_of_week", ""))
                for b in _preferences_bullets(llm_agent, day=day):
                    st.markdown(
                        f"<div style='padding:6px 0;border-top:1px solid #e5e7eb;color:#0f172a;font-size:13px'>• {b}</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # Full-width Action History section (below both columns).
    action_card = st.container(border=True)
    with action_card:
        st.subheader("Action History")
        st.caption("A full log of what the agent did this episode")
        traj = st.session_state.trajectory or []
        if not traj:
            st.info("Run an episode to populate action history.")
        else:
            table_html, tsv, total = _action_history_table_html(llm_agent, traj)
            st.markdown(table_html, unsafe_allow_html=True)
            _clipboard_export_button("Export Actions", tsv)

    st.divider()
    graph_card = st.container(border=True)
    with graph_card:
        st.subheader("Reward Graph")
        img_path = Path("training") / "reward_comparison.png"
        if img_path.exists():
            st.image(str(img_path), caption="Random vs Baseline vs LLM (moving averages)", use_container_width=True)
        else:
            st.info(
                "No reward comparison plot found at `training/reward_comparison.png`. "
                "Generate it by running `python training/evaluate_agents.py -n 100`."
            )


if __name__ == "__main__":
    main()

