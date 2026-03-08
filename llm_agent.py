"""
LLM-powered policy agent for LifeOps.

Primary inference: Groq-hosted LLMs (fast + strong).
Fallbacks:
  1) Local HuggingFace model (if available)
  2) Baseline heuristic agent

This module is intentionally lightweight and avoids RL frameworks.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Hackathon-friendly: allow `python llm_agent.py` from repo root.
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.actions import Action, ActionType  # noqa: E402
from env.actions import mask_illegal_actions  # noqa: E402
from env.lifeops_env import LifeOpsEnv  # noqa: E402
from env.baseline_agent import choose_baseline_action  # noqa: E402
from env.reward import detect_overlaps, travel_issues  # noqa: E402


ALLOWED_ACTION_TYPES: Tuple[str, ...] = tuple(a.value for a in ActionType)


def _min_to_hhmm(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h:02d}:{mm:02d}"


def _summarize_calendar(calendar: List[Dict[str, Any]], max_items: int = 12) -> str:
    if not calendar:
        return "No existing events."
    ordered = sorted(calendar, key=lambda e: (int(e.get("start_min", 0)), int(e.get("end_min", 0))))
    lines: List[str] = []
    for e in ordered[:max_items]:
        start = _min_to_hhmm(int(e.get("start_min", 0)))
        end = _min_to_hhmm(int(e.get("end_min", 0)))
        title = str(e.get("title", "Untitled"))
        loc = str(e.get("location", "Unknown"))
        kind = str(e.get("kind", "meeting"))
        imp = int(e.get("importance", 1))
        lines.append(f"- {start}-{end} | {title} | {loc} | kind={kind} | importance={imp}")
    if len(ordered) > max_items:
        lines.append(f"- ... ({len(ordered) - max_items} more)")
    return "\n".join(lines)


def _describe_request(req: Optional[Dict[str, Any]]) -> str:
    if req is None:
        return "No current incoming request."
    start = _min_to_hhmm(int(req.get("start_min", 0)))
    end = _min_to_hhmm(int(req.get("end_min", 0)))
    title = str(req.get("title", "Untitled"))
    loc = str(req.get("location", "Unknown"))
    imp = int(req.get("importance", 1))
    flexible = bool(req.get("flexible", True))
    from_person = str(req.get("from_person", "Unknown"))
    return f"{title} ({start}-{end}) @ {loc} | importance={imp} | flexible={flexible} | from={from_person}"


def _summarize_valid_actions(valid_actions: Sequence[Action], max_per_type: int = 3) -> str:
    """
    Summarize concrete valid actions without forcing the model to output parameters.

    The model is still required to output ONLY the action_type string.
    """

    by_type: Dict[str, List[Action]] = {}
    for a in valid_actions:
        by_type.setdefault(a.action_type.value, []).append(a)

    lines: List[str] = []
    for at in ALLOWED_ACTION_TYPES:
        actions = by_type.get(at, [])
        if not actions:
            continue
        if at == ActionType.block_focus_time.value:
            opts = []
            for a in actions[:max_per_type]:
                s = _min_to_hhmm(int(a.new_start_min or 0))
                d = int(a.duration_min or 0)
                opts.append(f"{s} for {d}m")
            lines.append(f"- {at} (examples: {', '.join(opts)})")
        elif at in {ActionType.reschedule_event.value, ActionType.propose_new_time.value}:
            opts = []
            for a in actions[:max_per_type]:
                s = _min_to_hhmm(int(a.new_start_min or 0))
                opts.append(s)
            lines.append(f"- {at} (examples start times: {', '.join(opts)})")
        else:
            lines.append(f"- {at}")
    return "\n".join(lines) if lines else "- (no valid actions)"


def summarize_state_for_llm(state: dict) -> str:
    """
    Convert the raw env `state` dict into a short natural-language summary.

    Why this helps:
    - Raw JSON/dicts are noisy (lots of keys, braces, repeated structure).
    - Smaller / faster models often reason better over a compact narrative:
      "what's on the calendar", "what's the request", "what free time exists".
    - We intentionally cap the amount of detail to keep the prompt short
      (< ~300 tokens) and to reduce distraction.
    """

    def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start < b_end and b_start < a_end

    def _travel_minutes(a_loc: str, b_loc: str) -> int:
        if a_loc == b_loc:
            return 0
        tt = state.get("travel_times", {}) or {}
        return int(tt.get(a_loc, {}).get(b_loc, 30))

    calendar = list(state.get("calendar", []))
    req = state.get("current_request")

    lines: List[str] = []

    # Calendar summary: show up to N upcoming events (time + title).
    lines.append("Calendar today:")
    if not calendar:
        lines.append("No scheduled events yet.")
    else:
        ordered = sorted(calendar, key=lambda e: (int(e.get("start_min", 0)), int(e.get("end_min", 0))))
        max_events = 8
        for e in ordered[:max_events]:
            start = _min_to_hhmm(int(e.get("start_min", 0)))
            title = str(e.get("title", "Untitled"))
            loc = str(e.get("location", "Unknown"))
            kind = str(e.get("kind", "meeting"))
            imp = int(e.get("importance", 1))
            # Highlight high-priority obligations/important meetings inline.
            tag = " [IMPORTANT]" if (imp >= 3 or kind == "obligation") else ""
            lines.append(f"{start} {title} ({loc}){tag}")
        if len(ordered) > max_events:
            lines.append(f"... plus {len(ordered) - max_events} more events")

        # Optional: include a tiny "free windows" hint.
        # We approximate the day as 08:00-20:00 for readability.
        day_start = 8 * 60
        day_end = 20 * 60
        free_blocks: List[Tuple[int, int]] = []
        cursor = day_start
        for e in ordered:
            s = max(day_start, int(e.get("start_min", 0)))
            if s > cursor:
                free_blocks.append((cursor, min(s, day_end)))
            cursor = max(cursor, int(e.get("end_min", 0)))
            if cursor >= day_end:
                break
        if cursor < day_end:
            free_blocks.append((cursor, day_end))

        # Prefer windows that can fit the request duration (if any); otherwise >=30m.
        req_dur = None
        if req is not None:
            req_dur = max(0, int(req.get("end_min", 0)) - int(req.get("start_min", 0)))

        min_free = max(30, int(req_dur or 0))
        free_blocks = [(s, e) for (s, e) in free_blocks if e - s >= min_free]
        if free_blocks:
            blocks = ", ".join(f"{_min_to_hhmm(s)}-{_min_to_hhmm(e)}" for s, e in free_blocks[:2])
            lines.append(f"Free windows (fit request): {blocks}")

    lines.append("")
    lines.append("Incoming request:")
    if req is None:
        lines.append("No incoming request.")
    else:
        start = _min_to_hhmm(int(req.get("start_min", 0)))
        end = _min_to_hhmm(int(req.get("end_min", 0)))
        title = str(req.get("title", "Untitled"))
        loc = str(req.get("location", "Unknown"))
        imp = int(req.get("importance", 1))
        flexible = bool(req.get("flexible", True))
        rs = int(req.get("start_min", 0))
        re_ = int(req.get("end_min", 0))
        lines.append(f"{title} at {start}-{end} ({loc}), importance={imp}, flexible={flexible}")

        # Highlight conflicts with existing events (this is the key feasibility constraint).
        conflicts: List[str] = []
        for e in calendar:
            if _overlap(rs, re_, int(e.get("start_min", 0)), int(e.get("end_min", 0))):
                conflicts.append(str(e.get("title", e.get("event_id", "event"))))
        if conflicts:
            lines.append(f"Conflicts with: {', '.join(conflicts[:3])}")
        else:
            lines.append("No time conflict with existing events.")

        # Highlight travel tightness around the request time (simple, local check).
        ordered = sorted(calendar, key=lambda e: (int(e.get("start_min", 0)), int(e.get("end_min", 0))))
        prev = None
        nxt = None
        for e in ordered:
            if int(e.get("end_min", 0)) <= rs:
                prev = e
            if nxt is None and int(e.get("start_min", 0)) >= re_:
                nxt = e
                break

        travel_notes: List[str] = []
        if prev is not None:
            avail = rs - int(prev.get("end_min", 0))
            need = _travel_minutes(str(prev.get("location", "Unknown")), loc)
            if need > avail:
                travel_notes.append(f"prev→request travel impossible (need {need}m, have {max(avail, 0)}m)")
        if nxt is not None:
            avail = int(nxt.get("start_min", 0)) - re_
            need = _travel_minutes(loc, str(nxt.get("location", "Unknown")))
            if need > avail:
                travel_notes.append(f"request→next travel impossible (need {need}m, have {max(avail, 0)}m)")
        if travel_notes:
            lines.append("Travel issue: " + "; ".join(travel_notes[:2]))

    return "\n".join(lines).strip()


def build_prompt(state: Dict[str, Any], valid_actions: Sequence[Action]) -> str:
    """
    Build an instruction prompt for an LLM.

    Requirements satisfied:
    - calendar summary
    - request description
    - list of valid actions
    - model must output ONLY one of the allowed action types
    """

    # Keep prompt short: a compact state summary + strict action instruction.
    # This tends to improve decision quality compared to dumping raw JSON.
    state_summary = summarize_state_for_llm(state)

    # Short hint about what's actually valid *right now* (types only).
    valid_types = sorted({a.action_type.value for a in valid_actions})
    valid_types_line = ", ".join(valid_types) if valid_types else "(none)"

    # Few-shot examples: minimal demonstrations to anchor the policy.
    # Kept very short to stay under the ~350 token budget.
    examples = """Example 1
Calendar today: 10:00 Team meeting; 13:00 Lunch; 15:00 Free
Incoming request: Meeting at 10:00
Correct action: reschedule_event

Example 2
Calendar today: 09:00 Free; 11:00 Project discussion
Incoming request: Meeting at 09:00
Correct action: accept_event
"""

    goal = (
        "Your goal is to maximize scheduling reward. "
        "Avoid conflicts, respect user preferences, and allocate time to important obligations."
    )

    prompt = f"""{examples}

{goal}

{state_summary}

Valid action types right now: {valid_types_line}

First analyze the situation.

For each action consider:
accept_event
reject_event
reschedule_event
propose_new_time
block_focus_time

For each action ask:
- Would this create a scheduling conflict?
- Would this respect user preferences?
- Would this improve the schedule?

Then choose the action with the best outcome.

Choose ONE action from:
{chr(10).join(ALLOWED_ACTION_TYPES)}

Output ONLY the action. Do not output explanations.
"""
    return prompt


@dataclass
class EpisodeMemory:
    """
    Minimal episode memory record used to improve next-episode prompting.
    """

    total_reward: float
    decisions: List[str]
    most_violated_constraint: str


def _compact_one_line(text: str) -> str:
    """Turn a multiline summary into a single compact line."""
    return " ".join(part.strip() for part in text.replace("\n", " ").split() if part.strip())


def _bucket_minute(start_min: int, bucket_minutes: int = 60) -> int:
    """Bucket a time into coarse slots for stable preference learning."""
    b = max(1, int(bucket_minutes))
    return (int(start_min) // b) * b


def _fmt_bucket(bmin: int) -> str:
    return _min_to_hhmm(int(bmin))


def parse_action_type(model_output: str) -> Optional[str]:
    """
    Safety parsing: extract an allowed action type from the model output.
    Returns None if no valid action type can be found.
    """

    if not model_output:
        return None
    text = model_output.strip().lower()

    # Exact match first.
    if text in ALLOWED_ACTION_TYPES:
        return text

    # Otherwise search within the output (e.g., "I choose accept_event.").
    for at in ALLOWED_ACTION_TYPES:
        if re.search(rf"\b{re.escape(at)}\b", text):
            return at
    return None


def choose_action_for_type(
    action_type: str,
    state: Dict[str, Any],
    valid_actions: Sequence[Action],
) -> Optional[Action]:
    """
    Map an action_type string to a concrete Action instance from valid_actions.

    Since the model outputs only the action type (no parameters), we pick a
    sensible member of that action type:
    - accept/reject: take the first
    - reschedule/propose: pick the option that minimizes overlaps/travel issues
    - block_focus_time: pick the option that minimizes overlaps/travel issues
    """

    candidates = [a for a in valid_actions if a.action_type.value == action_type]
    if not candidates:
        return None
    if action_type in {ActionType.accept_event.value, ActionType.reject_event.value}:
        return candidates[0]

    calendar = list(state.get("calendar", []))
    travel = state.get("travel_times", {})

    def score(a: Action) -> Tuple[int, int]:
        # Smaller is better: (overlap_count, travel_issue_count)
        if a.action_type == ActionType.block_focus_time:
            start = int(a.new_start_min or 0)
            dur = int(a.duration_min or 0)
            added = {
                "event_id": "focus_sim",
                "start_min": start,
                "end_min": start + dur,
                "location": state.get("persona", {}).get("primary_work_location", "Home"),
            }
            sim = calendar + [added]
        else:
            # reschedule/propose
            req = state.get("current_request") or {}
            added = dict(req)
            added["start_min"] = int(a.new_start_min or added.get("start_min", 0))
            added["end_min"] = int(a.new_end_min or added.get("end_min", 0))
            # propose_new_time does NOT schedule in env semantics, so don't add it.
            sim = calendar if a.action_type == ActionType.propose_new_time else calendar + [added]

        return (len(detect_overlaps(sim)), len(travel_issues(sim, travel)))

    candidates.sort(key=score)
    return candidates[0]


def call_groq_llm(prompt: str) -> str:
    """
    Call Groq-hosted LLM and return the raw text output.

    Primary model:
      meta-llama/llama-4-scout-17b-16e-instruct
    Fallback model:
      llama-3.3-70b-versatile
    """

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Groq SDK not available: {e}") from e

    client = Groq(api_key=api_key)

    def _call(model: str) -> str:
        # Groq SDK is OpenAI-like; keep extraction robust.
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only the requested action string."},
                {"role": "user", "content": prompt},
            ],
            # Deterministic, decision-focused decoding.
            temperature=0.1,
            top_p=1.0,
            max_tokens=5,
        )
        # Most common shape: resp.choices[0].message.content
        content = None
        try:
            content = resp.choices[0].message.content
        except Exception:
            pass
        if content is None:
            # Try dict-like fallback.
            content = getattr(resp, "output_text", None) or str(resp)
        return str(content).strip()

    # Allow override for evaluation scripts / experiments.
    # Defaults remain the same to avoid breaking existing behavior.
    primary = os.environ.get("LIFEOPS_GROQ_PRIMARY_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
    secondary = os.environ.get("LIFEOPS_GROQ_FALLBACK_MODEL") or "llama-3.3-70b-versatile"

    try:
        return _call(primary)
    except Exception:
        return _call(secondary)


@dataclass
class LLMAgent:
    """
    Policy wrapper that chooses actions for LifeOpsEnv.

    Behavior:
    - Try Groq (fast, strong)
    - If Groq fails, try local HF model (if configured and available)
    - If local fails, fall back to baseline heuristic
    """

    local_model_name: Optional[str] = None  # e.g. "gpt2" or a local path
    # Deterministic, decision-focused decoding.
    local_max_new_tokens: int = 5
    local_temperature: float = 0.1
    local_top_p: float = 1.0

    _hf_model: Any = None
    _hf_tokenizer: Any = None

    # Cross-episode memory (prompt context).
    episode_memory: List[EpisodeMemory] = None  # type: ignore[assignment]
    top_success_examples: List[str] = None  # type: ignore[assignment]
    # Persistent user preference model across episodes.
    # hour_bucket_minute -> {"accepted": int, "rejected": int, "good": int, "bad": int}
    preference_model: Dict[int, Dict[str, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Keep attributes mutable but initialized.
        if self.episode_memory is None:
            self.episode_memory = []
        if self.top_success_examples is None:
            self.top_success_examples = []
        if self.preference_model is None:
            self.preference_model = {}

    def _update_preference(self, bucket_min: int, *, accepted: bool, reward: float) -> None:
        row = self.preference_model.get(bucket_min)
        if row is None:
            row = {"accepted": 0, "rejected": 0, "good": 0, "bad": 0}
            self.preference_model[bucket_min] = row
        if accepted:
            row["accepted"] += 1
        else:
            row["rejected"] += 1
        if float(reward) >= 0.0:
            row["good"] += 1
        else:
            row["bad"] += 1

    def _preference_score(self, bucket_min: int) -> float:
        r = self.preference_model.get(bucket_min) or {"accepted": 0, "rejected": 0, "good": 0, "bad": 0}
        return float(r["accepted"] - r["rejected"]) + 0.5 * float(r["good"] - r["bad"])

    def summarize_preference_model_for_prompt(self, state: Dict[str, Any]) -> str:
        """
        Compact preference summary injected into every prompt.

        This is the simplest form of "learning across episodes" without changing
        the environment or reward: the LLM sees which time buckets historically
        produced better outcomes and can avoid repeating bad choices.
        """
        if not self.preference_model:
            return "Learned time preferences: (no history yet)"

        scored: List[Tuple[int, float, int]] = []
        for b, row in self.preference_model.items():
            seen = int(row.get("accepted", 0) + row.get("rejected", 0))
            if seen < 2:
                continue
            scored.append((int(b), self._preference_score(int(b)), seen))
        if not scored:
            return "Learned time preferences: (not enough history yet)"

        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[:3]
        bottom = sorted(scored, key=lambda t: t[1])[:3]

        def fmt(items: List[Tuple[int, float, int]]) -> str:
            return ", ".join(f"{_fmt_bucket(b)}({s:+.1f}/{seen}x)" for b, s, seen in items)

        req = state.get("current_request")
        if req is not None:
            rb = _bucket_minute(int(req.get("start_min", 0)), 60)
            rs = self._preference_score(rb)
            rrow = self.preference_model.get(rb) or {}
            seen = int(rrow.get("accepted", 0) + rrow.get("rejected", 0))
            current = f"current={_fmt_bucket(rb)} score={rs:+.1f} seen={seen}x"
        else:
            current = "current=(none)"

        return f"Learned time preferences (hour buckets): prefer {fmt(top)} | avoid {fmt(bottom)} | {current}"

    def _ensure_local_model(self) -> None:
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return
        if not self.local_model_name:
            raise RuntimeError("No local_model_name configured")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers not available: {e}") from e

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
        self._hf_model = AutoModelForCausalLM.from_pretrained(self.local_model_name)

    def call_local_hf_llm(self, prompt: str) -> str:
        """
        Local HuggingFace fallback inference.
        """

        self._ensure_local_model()
        tok = self._hf_tokenizer
        model = self._hf_model

        # Use model.generate() (deterministic settings) as requested.
        inputs = tok(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(self.local_max_new_tokens),
            temperature=float(self.local_temperature),
            top_p=float(self.local_top_p),
            do_sample=False,
        )
        # Decode only the newly generated tokens to avoid echoing the prompt.
        gen_ids = output_ids[0][input_len:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    def choose_action(self, env: LifeOpsEnv) -> Action:
        """
        Choose a concrete valid action for the given environment state.

        Pipeline:
          try: Groq
          except: local HF
          except: baseline heuristic
        """

        state = env.observation()
        valid_actions = env.valid_actions()
        valid_actions = mask_illegal_actions(state, valid_actions)
        if not valid_actions:
            raise RuntimeError("No valid actions available")

        prompt = self.build_prompt_with_memory(state, valid_actions)

        # 1) Groq inference
        try:
            raw = call_groq_llm(prompt)
            at = parse_action_type(raw)
            if at:
                chosen = choose_action_for_type(at, state, valid_actions)
                if chosen is not None:
                    return chosen
        except Exception:
            pass

        # 2) Local HF fallback
        try:
            raw = self.call_local_hf_llm(prompt)
            at = parse_action_type(raw)
            if at:
                chosen = choose_action_for_type(at, state, valid_actions)
                if chosen is not None:
                    return chosen
        except Exception:
            pass

        # 3) Baseline heuristic fallback (guaranteed valid Action)
        try:
            return self.choose_preference_aware_action(state, valid_actions)
        except Exception:
            return choose_baseline_action(state, valid_actions)

    def choose_preference_aware_action(self, state: Dict[str, Any], valid_actions: Sequence[Action]) -> Action:
        """
        Preference-aware fallback policy (no LLM call).

        This ensures the "LLM agent" can still improve across episodes even if
        the external LLM is unavailable, by exploiting the learned preference
        model (accepted/rejected time buckets).

        Random agent does NOT have access to this, so the gap is visible.
        """

        actions = list(valid_actions)
        if not actions:
            raise RuntimeError("No valid actions")

        req = state.get("current_request")
        if req is None:
            # No request: prefer late productivity focus blocks when tasks remain.
            tasks = state.get("tasks", []) or []
            has_work = any(int(t.get("remaining_minutes", 0)) > 0 for t in tasks)
            if has_work:
                late_focus = [a for a in actions if a.action_type == ActionType.block_focus_time and int(a.new_start_min or 0) >= 20 * 60]
                if late_focus:
                    # Choose the one with least feasibility risk (already masked) and fixed order.
                    late_focus.sort(key=lambda a: (int(a.new_start_min or 0), int(a.duration_min or 0)))
                    return late_focus[0]
            # Otherwise choose any focus action, else first action.
            focus = [a for a in actions if a.action_type == ActionType.block_focus_time]
            if focus:
                focus.sort(key=lambda a: (int(a.new_start_min or 0), int(a.duration_min or 0)))
                return focus[0]
            return actions[0]

        bucket = _bucket_minute(int(req.get("start_min", 0)), 60)
        score = self._preference_score(bucket)

        # Helper: pick best reschedule/propose candidate according to learned preference score.
        def best_time_shift(action_type: ActionType) -> Optional[Action]:
            cands = [a for a in actions if a.action_type == action_type]
            if not cands:
                return None
            cands.sort(
                key=lambda a: (
                    -self._preference_score(_bucket_minute(int(a.new_start_min or 0), 60)),
                    int(a.new_start_min or 0),
                )
            )
            return cands[0]

        accept = next((a for a in actions if a.action_type == ActionType.accept_event), None)
        reject = next((a for a in actions if a.action_type == ActionType.reject_event), None)

        # Strongly preferred buckets: accept if feasible.
        if score >= 0.5 and accept is not None:
            return accept

        # Strongly avoided buckets: try to move it, otherwise reject.
        if score <= -0.5:
            res = best_time_shift(ActionType.reschedule_event)
            if res is not None:
                return res
            prop = best_time_shift(ActionType.propose_new_time)
            if prop is not None:
                return prop
            if reject is not None:
                return reject

        # Unknown/neutral: prefer rescheduling into a preferred bucket if available; otherwise accept.
        res = best_time_shift(ActionType.reschedule_event)
        if res is not None:
            # Only do this if it moves into a meaningfully better bucket.
            new_bucket = _bucket_minute(int(res.new_start_min or 0), 60)
            if self._preference_score(new_bucket) > score + 0.5:
                return res
        if accept is not None:
            return accept
        # Final fallback.
        return reject if reject is not None else actions[0]

    def build_prompt_with_memory(self, state: Dict[str, Any], valid_actions: Sequence[Action]) -> str:
        """
        Prepend cross-episode memory to the standard prompt.

        This gives the LLM a tiny feedback loop without changing the overall
        architecture: it can reuse patterns that led to higher reward in the
        recent past.
        """

        memory_lines: List[str] = []

        # Persistent preference model is injected into every prompt so the LLM
        # can learn from prior outcomes across episodes (without changing env code).
        memory_lines.append(self.summarize_preference_model_for_prompt(state))
        memory_lines.append(
            "Checklist: consult preferences FIRST. If the request time bucket is in 'avoid', "
            "prefer reschedule_event/propose_new_time/reject_event. If in 'prefer', accept_event if feasible."
        )
        if self.episode_memory:
            memory_lines.append("Recent episode feedback (to improve reward):")
            for m in self.episode_memory[-3:]:
                dec = ", ".join(m.decisions[:6]) + ("..." if len(m.decisions) > 6 else "")
                memory_lines.append(
                    f"- total_reward={m.total_reward:+.2f} | decisions=[{dec}] | most_violated={m.most_violated_constraint}"
                )

        if self.top_success_examples:
            memory_lines.append("")
            memory_lines.append("Top successful past decisions (examples):")
            for ex in self.top_success_examples[:3]:
                memory_lines.append(ex)

        prefix = ("\n".join(memory_lines).strip() + "\n\n") if memory_lines else ""
        return prefix + build_prompt(state, valid_actions)

    def on_episode_end(self, trajectory: List[Dict[str, Any]], total_reward: float) -> None:
        """
        Update memory after an episode.

        Expected trajectory format (as produced by training/train_rl.py):
          [{"obs": ..., "action": ..., "reward": ..., "info": ...}, ...]
        """

        decisions = [str(t.get("action", {}).get("action_type", "?")) for t in trajectory]

        # Count which constraints were violated most.
        overlap_count = 0
        travel_count = 0
        pref_count = 0
        blocked_count = 0  # best-effort; depends on masking, should usually be 0
        for t in trajectory:
            info = t.get("info", {}) or {}
            overlap_count += 1 if info.get("overlaps") else 0
            travel_count += 1 if info.get("travel_issues") else 0
            rb = (info.get("reward_breakdown") or {})
            if rb.get("preference_penalty"):
                pref_count += 1
            if rb.get("blocked_hours_penalty"):
                blocked_count += 1

        counts = {
            "double_booking": overlap_count,
            "travel": travel_count,
            "preference": pref_count,
            "productivity_window": blocked_count,
        }
        most_violated = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "none"

        self.episode_memory.append(
            EpisodeMemory(
                total_reward=float(total_reward),
                decisions=decisions,
                most_violated_constraint=most_violated,
            )
        )
        # Keep memory bounded.
        if len(self.episode_memory) > 20:
            self.episode_memory = self.episode_memory[-20:]

        # Update top successful decisions as few-shot examples.
        # Pick up to 3 steps with highest reward from this episode and store compact examples.
        steps_sorted = sorted(trajectory, key=lambda t: float(t.get("reward", 0.0)), reverse=True)
        for t in steps_sorted[:3]:
            obs = t.get("obs") or {}
            act = t.get("action") or {}
            at = str(act.get("action_type", "")).strip()
            if at not in ALLOWED_ACTION_TYPES:
                continue
            # Use existing state summarizer, but compress heavily.
            s = _compact_one_line(summarize_state_for_llm(obs))
            ex = f"- {s} => Correct action: {at}"
            self.top_success_examples.insert(0, ex)

        # Keep examples bounded + unique-ish.
        dedup: List[str] = []
        seen = set()
        for ex in self.top_success_examples:
            if ex in seen:
                continue
            seen.add(ex)
            dedup.append(ex)
            if len(dedup) >= 10:
                break
        self.top_success_examples = dedup

        # Update persistent preference model:
        # Track which hour buckets tend to lead to good outcomes when accepted vs rejected.
        for t in trajectory:
            obs = t.get("obs") or {}
            act = t.get("action") or {}
            at = str(act.get("action_type", "")).strip()
            r = float(t.get("reward", 0.0))

            req = obs.get("current_request")
            if req is None:
                continue

            if at in {"accept_event", "reject_event"}:
                start_min = int(req.get("start_min", 0))
            elif at in {"reschedule_event", "propose_new_time"}:
                start_min = int(act.get("new_start_min") or req.get("start_min", 0))
            else:
                continue

            bucket = _bucket_minute(start_min, 60)
            accepted = at in {"accept_event", "reschedule_event"}
            rejected = at == "reject_event"
            if accepted or rejected:
                self._update_preference(bucket, accepted=accepted, reward=r)


if __name__ == "__main__":
    # Quick smoke demo (uses baseline if no GROQ_API_KEY / local model configured).
    env = LifeOpsEnv(seed=0)
    env.reset()
    agent = LLMAgent(local_model_name=os.environ.get("LIFEOPS_LOCAL_MODEL"))
    a = agent.choose_action(env)
    print(a.to_dict())

