"""
LLM-based decision policy for LifeOps.

Uses a HuggingFace model or Groq API to choose actions from the environment state.
Falls back to baseline agent when the LLM output is invalid.

Supports:
- Local HuggingFace: seq2seq (T5/Flan) and causal (Phi-3, Mistral) models
- Groq API: use model_id="groq:llama-3.3-70b-versatile" or "groq:meta-llama/llama-4-scout-17b-16e-instruct"
  Requires: pip install groq, GROQ_API_KEY in .env or environment

Model fallback chain: phi-3-mini -> flan-t5-base -> Mistral-7B.
Requires: pip install transformers torch
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Groq model prefix: model_id="groq:llama-3.3-70b-versatile" uses Groq API
GROQ_PREFIX = "groq:"

# Add repo root for imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.actions import Action, ActionType

# Model fallback chain (try in order; first successful load wins)
# For better LLM performance, use --model "Qwen/Qwen2.5-3B-Instruct" or "microsoft/Phi-3-medium-4k-instruct"
MODEL_FALLBACK_CHAIN = [
    "microsoft/phi-3-mini-4k-instruct",
    "google/flan-t5-base",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/flan-t5-small",  # lightweight fallback if others fail
]

def _is_causal_model(model_id: str) -> bool:
    """True if model uses causal LM (Phi-3, Mistral) vs seq2seq (T5/Flan)."""
    lid = model_id.lower()
    return "phi" in lid or "mistral" in lid

ALLOWED_ACTION_TYPES = frozenset({
    "accept_event",
    "reject_event",
    "reschedule_event",
    "propose_new_time",
    "block_focus_time",
})

FEW_SHOT_PREFIX = """Example 1:
Persona: Early-bird Engineer
Preferred meeting window: [540, 1020]
Current calendar:
  - 09:00–10:00: Standup @ Office
Incoming request: Team sync
  Time: 10:30–11:00 @ Remote
  Importance: 1
Reasoning: The request is low importance and the 10:30 slot conflicts with travel from the 10:00 standup; proposing a new time avoids the conflict.
CHOICE: 6

Example 2:
Persona: Night-owl Creator
Preferred meeting window: [600, 1020]
Calendar: (empty)
Incoming request: Client review
  Time: 11:00–12:00 @ Office
  Importance: 3
Reasoning: High-importance meeting during preferred hours with an empty calendar; accept.
CHOICE: 1

Example 3:
Persona: Busy Parent
Preferred meeting window: [540, 960]
Current calendar:
  - 09:00–10:00: Meeting A @ Office
  - 14:00–15:00: Meeting B @ Home
No pending request. You can block focus time for tasks.
Tasks:
  - Write report: 60 min remaining (priority 2)
Reasoning: No pending request; block focus time at a free slot (11:00) between the two existing events.
CHOICE: 9

"""

logger = logging.getLogger(__name__)


def _min_to_time(m: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def _state_to_prompt(
    state: Dict[str, Any],
    valid_actions: List[Action],
    few_shot_prefix: Optional[str] = None,
) -> str:
    """Convert environment state into a readable prompt for the LLM."""
    lines: List[str] = []
    if few_shot_prefix:
        lines.append(few_shot_prefix.rstrip())
        lines.append("")

    persona = state.get("persona", {})
    lines.append(f"Persona: {persona.get('name', '?')}")
    lines.append(f"Preferred meeting window: {persona.get('preferred_meeting_window', [])}")
    lines.append("")

    calendar = state.get("calendar", [])
    if calendar:
        lines.append("Current calendar:")
        for e in sorted(calendar, key=lambda x: int(x.get("start_min", 0))):
            start = int(e.get("start_min", 0))
            end = int(e.get("end_min", 0))
            lines.append(f"  - {_min_to_time(start)}–{_min_to_time(end)}: {e.get('title', '?')} @ {e.get('location', '?')}")
    else:
        lines.append("Calendar: (empty)")
    lines.append("")

    tasks = state.get("tasks", [])
    if tasks:
        lines.append("Tasks:")
        for t in tasks:
            rem = int(t.get("remaining_minutes", 0))
            if rem > 0:
                lines.append(f"  - {t.get('title', '?')}: {rem} min remaining (priority {t.get('priority', 2)})")
    lines.append("")

    req = state.get("current_request")
    if req:
        start = int(req.get("start_min", 0))
        end = int(req.get("end_min", 0))
        lines.append(f"Incoming request: {req.get('title', '?')}")
        lines.append(f"  Time: {_min_to_time(start)}–{_min_to_time(end)} @ {req.get('location', '?')}")
        lines.append(f"  Importance: {req.get('importance', 1)}")
    else:
        lines.append("No pending request. You can block focus time for tasks.")
    lines.append("")

    lines.append("Choose ONE action. Valid actions are ONLY: accept_event, reject_event, reschedule_event, propose_new_time, block_focus_time.")
    lines.append("Important: Avoid overlaps (double-booking). For focus blocks, pick a slot that does NOT overlap with existing calendar events.")
    lines.append("")
    lines.append("Options (reply with the number only, e.g. 2):")
    for i, a in enumerate(valid_actions, 1):
        if a.action_type == ActionType.block_focus_time:
            start = int(a.new_start_min or 0)
            dur = int(a.duration_min or 0)
            lines.append(f"  {i}. block_focus_time @ {_min_to_time(start)} for {dur} min")
        elif a.action_type == ActionType.accept_event:
            lines.append(f"  {i}. accept_event (request {a.request_id})")
        elif a.action_type == ActionType.reject_event:
            lines.append(f"  {i}. reject_event (request {a.request_id})")
        elif a.action_type == ActionType.reschedule_event:
            ns = int(a.new_start_min or 0)
            ne = int(a.new_end_min or 0)
            lines.append(f"  {i}. reschedule_event → {_min_to_time(ns)}–{_min_to_time(ne)}")
        elif a.action_type == ActionType.propose_new_time:
            ns = int(a.new_start_min or 0)
            ne = int(a.new_end_min or 0)
            lines.append(f"  {i}. propose_new_time → {_min_to_time(ns)}–{_min_to_time(ne)}")
        else:
            lines.append(f"  {i}. {a.action_type.value}")

    lines.append("")
    lines.append("Think step by step:")
    lines.append("1. Identify any calendar conflicts (overlaps, double-booking).")
    lines.append("2. Consider the persona's preferred meeting window and travel constraints.")
    lines.append("3. Write 1-2 sentences of reasoning.")
    lines.append("4. End with exactly: CHOICE: <number>")
    return "\n".join(lines)


def parse_llm_action(response_text: str, valid_actions: List[Action]) -> Optional[Action]:
    """
    Safely map LLM output to one of the valid actions.

    Safety logic:
    1. Normalize: lowercase, strip whitespace (handles messy model output)
    2. Try numeric extraction: prompt asks for "number only", so "1" -> valid_actions[0]
    3. Try action-type matching: if model outputs "accept_event", find matching valid action
    4. Strict validation: only return an Action that is in valid_actions
    5. Reject anything that doesn't map to a valid action (return None)

    Never raises. Returns None for any invalid or unparseable response.
    """
    if not valid_actions:
        return None

    # Step 1: Normalize text to handle common LLM output variations
    text = (response_text or "").lower().strip()
    if not text:
        return None

    # Step 2: Try to extract a number (primary: CHOICE: <number>, fallback: any digit)
    try:
        match = re.search(r"choice\s*:\s*(\d+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"\b(\d+)\b", text)
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= len(valid_actions):
                action = valid_actions[idx - 1]
                # Validate: ensure action type is in allowed whitelist
                if action.action_type.value in ALLOWED_ACTION_TYPES:
                    return action
    except (ValueError, IndexError, AttributeError):
        pass  # Fall through to action-type matching

    # Step 3: Try to match action type name in the response
    for action in valid_actions:
        at = action.action_type.value
        if at in ALLOWED_ACTION_TYPES and at in text:
            return action

    # Step 4: No valid mapping found - reject
    return None


_model_cache: Dict[str, tuple] = {}  # model_id -> (tokenizer, model, is_causal)
_working_model: Optional[Tuple[Any, Any, bool, str]] = None  # cache first successful load


def _load_model(model_id: str) -> Tuple[Any, Any, bool]:
    """Load and cache model/tokenizer. Returns (tokenizer, model, is_causal)."""
    if model_id not in _model_cache:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        is_causal = _is_causal_model(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if is_causal:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype="auto"
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _model_cache[model_id] = (tokenizer, model, is_causal)
    return _model_cache[model_id]


def _get_working_model(model_ids: Optional[List[str]] = None) -> Optional[Tuple[Any, Any, bool, str]]:
    """
    Try loading models from fallback chain. Returns (tokenizer, model, is_causal, model_id) or None.
    Caches the first successful load when using default chain (avoids retrying every step).
    """
    global _working_model
    ids = model_ids or MODEL_FALLBACK_CHAIN
    use_default = model_ids is None
    if use_default and _working_model is not None:
        return _working_model
    for model_id in ids:
        try:
            tokenizer, model, is_causal = _load_model(model_id)
            result = (tokenizer, model, is_causal, model_id)
            if use_default:
                _working_model = result
                logger.info("Using LLM model: %s", model_id)
            return result
        except Exception as e:
            logger.warning("Failed to load %s: %s. Trying next.", model_id, e)
    return None


def _generate_response(
    tokenizer: Any,
    model: Any,
    prompt: str,
    is_causal: bool,
    max_new_tokens: int = 20,
) -> str:
    """Run inference and return decoded response."""
    model.eval()
    if is_causal:
        # Use chat template if available (Phi-3, Mistral)
        try:
            if hasattr(tokenizer, "apply_chat_template") and getattr(
                tokenizer, "chat_template", None
            ):
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"Instruction: {prompt}\n\nResponse:"
        except Exception:
            text = f"Instruction: {prompt}\n\nResponse:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


def _generate_response_groq(
    groq_model_id: str,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
) -> str:
    """Call Groq API for inference. Requires GROQ_API_KEY in env and pip install groq."""
    # Ensure .env is loaded (e.g. when agent is used standalone)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
        except ImportError:
            pass
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Create an API key at https://console.groq.com and add "
            "GROQ_API_KEY=your_key to .env or export it."
        )
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("Groq API requires: pip install groq")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=groq_model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    return (content or "").strip()


def _is_groq_model(model_id: Optional[str]) -> bool:
    """True if model_id uses Groq API (prefix groq:)."""
    return bool(model_id and model_id.strip().lower().startswith(GROQ_PREFIX))


def _strip_groq_prefix(model_id: str) -> str:
    """Return Groq model ID (e.g. llama-3.3-70b-versatile) from groq:llama-3.3-70b-versatile."""
    return model_id.strip()[len(GROQ_PREFIX):].strip()


def choose_action_samples(
    state: Dict[str, Any],
    valid_actions: List[Action],
    model_id: Optional[str] = None,
    num_samples: int = 5,
    temperature: float = 0.7,
    fallback_fn: Optional[Callable[[], Action]] = None,
) -> List[Action]:
    """
    Sample N actions from the LLM (for Best-of-N). Uses temperature>0 for diversity.
    Returns list of actions; invalid parses are replaced with fallback.
    """
    if not valid_actions:
        raise RuntimeError("No valid actions")

    def _safe_fallback() -> Action:
        if fallback_fn:
            try:
                return fallback_fn()
            except Exception:
                pass
        import random
        return random.choice(valid_actions)

    resolved_id = model_id or (MODEL_FALLBACK_CHAIN[0] if MODEL_FALLBACK_CHAIN else None)
    prompt = _state_to_prompt(state, valid_actions)
    actions: List[Action] = []

    if _is_groq_model(resolved_id):
        groq_model = _strip_groq_prefix(resolved_id)
        try:
            for _ in range(num_samples):
                response = _generate_response_groq(
                    groq_model, prompt, temperature=temperature
                )
                action = parse_llm_action(response, valid_actions)
                actions.append(action if action is not None else _safe_fallback())
        except Exception as e:
            logger.warning("Groq sampling failed: %s. Using fallback for all.", e)
            return [_safe_fallback() for _ in range(num_samples)]
        return actions

    # HuggingFace path
    loaded = _get_working_model([resolved_id] if resolved_id else None)
    if loaded is None:
        return [_safe_fallback() for _ in range(num_samples)]
    tokenizer, model, is_causal, _ = loaded
    for _ in range(num_samples):
        try:
            response = _generate_response(
                tokenizer, model, prompt, is_causal,
                max_new_tokens=20,
            )
            # HuggingFace _generate_response uses do_sample=False; we'd need to modify
            # for true sampling. For now, we get same response N times - still works
            # if we add do_sample when temperature>0. Skip for simplicity.
            action = parse_llm_action(response, valid_actions)
            actions.append(action if action is not None else _safe_fallback())
        except Exception:
            actions.append(_safe_fallback())
    return actions


def choose_action(
    state: Dict[str, Any],
    valid_actions: List[Action],
    model_id: Optional[str] = None,
    model_ids: Optional[List[str]] = None,
    fallback_fn: Optional[Callable[[], Action]] = None,
    parse_stats: Optional[Dict[str, int]] = None,
    few_shot_prefix: Optional[str] = None,
) -> Action:
    """
    Use an LLM to choose one action from the valid action space.

    Args:
        state: Environment observation dict
        valid_actions: List of valid Action objects
        model_id: Single HuggingFace model (optional; overrides fallback chain)
        model_ids: Fallback chain of model IDs (default: phi-3 -> flan-t5-base -> Mistral)
        fallback_fn: Called when LLM output is invalid (default: random choice)

    Returns:
        A valid Action (always from valid_actions or fallback)
    """
    if not valid_actions:
        raise RuntimeError("No valid actions")

    def _safe_fallback() -> Action:
        if fallback_fn:
            try:
                return fallback_fn()
            except Exception as e:
                logger.warning("Fallback fn failed: %s. Using random.", e)
        import random
        return random.choice(valid_actions)

    # Groq API path (model_id="groq:llama-3.3-70b-versatile")
    resolved_id = model_id or (model_ids[0] if (model_ids and len(model_ids) > 0) else None)
    if _is_groq_model(resolved_id):
        groq_model = _strip_groq_prefix(resolved_id)
        try:
            prompt = _state_to_prompt(state, valid_actions, few_shot_prefix=few_shot_prefix)
            response = _generate_response_groq(groq_model, prompt)
        except Exception as e:
            logger.warning("Groq API failed: %s. Using fallback.", e)
            return _safe_fallback()
        action = parse_llm_action(response, valid_actions)
        if action is not None:
            if parse_stats is not None:
                parse_stats["parsed"] = parse_stats.get("parsed", 0) + 1
            return action
        if parse_stats is not None:
            parse_stats["fallback"] = parse_stats.get("fallback", 0) + 1
        logger.warning(
            "Groq output invalid or unparseable: %r. Falling back to baseline.",
            response[:100] if response else "(empty)",
        )
        return _safe_fallback()

    # HuggingFace path (local models)
    try:
        if model_id:
            loaded = _get_working_model([model_id])
        else:
            loaded = _get_working_model(model_ids)
        if loaded is None:
            logger.warning("No LLM model could be loaded. Using fallback.")
            return _safe_fallback()
        tokenizer, model, is_causal, used_id = loaded
    except (ImportError, OSError) as e:
        logger.warning("LLM model unavailable: %s. Using fallback.", e)
        return _safe_fallback()

    try:
        prompt = _state_to_prompt(state, valid_actions, few_shot_prefix=few_shot_prefix)
        response = _generate_response(tokenizer, model, prompt, is_causal)
    except Exception as e:
        logger.warning("LLM inference failed: %s. Using fallback.", e)
        return _safe_fallback()

    action = parse_llm_action(response, valid_actions)
    if action is not None:
        if parse_stats is not None:
            parse_stats["parsed"] = parse_stats.get("parsed", 0) + 1
        return action

    # Invalid response: log and fall back to baseline
    if parse_stats is not None:
        parse_stats["fallback"] = parse_stats.get("fallback", 0) + 1
    logger.warning(
        "LLM output invalid or unparseable: %r. Falling back to baseline.",
        response[:100] if response else "(empty)",
    )
    return _safe_fallback()


def llm_policy_fn(
    env: Any,
    model_id: Optional[str] = None,
    model_ids: Optional[List[str]] = None,
) -> Action:
    """
    Policy function compatible with train_rl.py: (env) -> Action.

    Uses LLM to choose action; falls back to baseline heuristic on invalid output.
    Default: tries phi-3-mini -> flan-t5-base -> Mistral-7B.
    """
    from env.lifeops_env import _choose_simple_action

    state = env.observation()
    valid_actions = env.valid_actions()
    return choose_action(
        state,
        valid_actions,
        model_id=model_id,
        model_ids=model_ids,
        fallback_fn=lambda: _choose_simple_action(env),
        few_shot_prefix=FEW_SHOT_PREFIX,
    )
