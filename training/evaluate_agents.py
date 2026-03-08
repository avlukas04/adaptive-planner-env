#!/usr/bin/env python3
"""
Evaluate and compare agent performance on the LifeOps environment.

Agents compared:
- random agent
- baseline heuristic agent
- LLM agent (LLMAgent.choose_action from llm_agent.py)

Example:
  python training/evaluate_agents.py --model groq:llama-3.3-70b-versatile -n 20

Notes:
- For a fair comparison, we run each agent on the SAME scenario sequence.
- Plot is both shown (plt.show) and saved to training/reward_comparison.png.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# Allow running as a script from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.lifeops_env import LifeOpsEnv  # noqa: E402
from env.actions import mask_illegal_actions  # noqa: E402
from env.scenario_generator import scenario_ids_by_difficulty  # noqa: E402
from env.baseline_agent import choose_baseline_action  # noqa: E402
from llm_agent import LLMAgent  # noqa: E402


def _moving_average(values: Sequence[float], window: int = 10) -> List[float]:
    if window <= 1:
        return [float(v) for v in values]
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += float(v)
        if i >= window:
            s -= float(values[i - window])
        out.append(s / float(min(i + 1, window)))
    return out


def _count_blocked_meeting_violations(calendar: List[Dict], blocked: Sequence[Tuple[int, int]]) -> int:
    """
    Count scheduled meeting-like events inside blocked hours.
    With action masking enabled, this should usually be 0 (but we still track it).
    """

    def overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
        return a_s < b_e and b_s < a_e

    count = 0
    for e in calendar:
        kind = str(e.get("kind", "meeting"))
        if kind not in {"meeting", "obligation", "personal"}:
            continue
        s = int(e.get("start_min", 0))
        en = int(e.get("end_min", 0))
        for bs, be in blocked:
            if overlaps(s, en, bs, be):
                count += 1
                break
    return count


def run_episode(
    env: LifeOpsEnv,
    policy_fn: Callable[[LifeOpsEnv], object],
    scenario_id: str,
    *,
    collect_trajectory: bool = False,
) -> Tuple[float, Dict[str, int], Optional[List[Dict]]]:
    """
    Reset the env to a fixed scenario and run until done.
    Returns:
      total episode reward,
      violation counts (travel/double_booking/goal_miss/productivity_window/preference),
      optional trajectory (for LLM memory).
    """

    obs = env.reset(scenario_id=scenario_id)
    total = 0.0
    done = False

    # Safety cap: should never be needed, but avoids infinite loops if buggy.
    cap = 200
    steps = 0

    traj: List[Dict] = []
    overlap_steps = 0
    travel_steps = 0
    preference_steps = 0

    while not done and steps < cap:
        steps += 1
        prev_obs = obs
        action = policy_fn(env)
        obs, reward, done, info = env.step(action)
        total += float(reward)

        if info.get("overlaps"):
            overlap_steps += 1
        if info.get("travel_issues"):
            travel_steps += 1
        rb = (info.get("reward_breakdown") or {})
        if rb.get("preference_penalty"):
            preference_steps += 1

        if collect_trajectory:
            traj.append(
                {
                    "obs": prev_obs,
                    "action": action.to_dict() if hasattr(action, "to_dict") else action,
                    "reward": float(reward),
                    "info": info,
                }
            )

    final = env.observation()
    goal_miss = 1 if any(int(t.get("remaining_minutes", 0)) > 0 for t in final.get("tasks", []) or []) else 0
    blocked = ((0, 9 * 60), (20 * 60, 23 * 60))
    productivity_window = _count_blocked_meeting_violations(final.get("calendar", []) or [], blocked)

    violations = {
        "double_booking": overlap_steps,
        "travel": travel_steps,
        "preference": preference_steps,
        "goal_miss": goal_miss,
        "productivity_window": productivity_window,
    }
    return total, violations, traj if collect_trajectory else None


def make_random_policy(seed: int) -> Callable[[LifeOpsEnv], object]:
    rng = random.Random(seed)

    def _policy(env: LifeOpsEnv) -> object:
        state = env.observation()
        valid = mask_illegal_actions(state, env.valid_actions())
        if not valid:
            raise RuntimeError("No valid actions")
        return rng.choice(valid)

    return _policy


def make_baseline_policy() -> Callable[[LifeOpsEnv], object]:
    def _policy(env: LifeOpsEnv) -> object:
        state = env.observation()
        valid = env.valid_actions()
        return choose_baseline_action(state, valid)

    return _policy


def make_llm_policy(agent: LLMAgent) -> Callable[[LifeOpsEnv], object]:
    def _policy(env: LifeOpsEnv) -> object:
        # LLMAgent internally falls back to local HF, then baseline.
        return agent.choose_action(env)

    return _policy


def parse_model_arg(model: str) -> None:
    """
    Configure the LLM selection.

    Supported:
    - groq:<model_name>  (e.g., groq:llama-3.3-70b-versatile)

    This sets env vars read by llm_agent.call_groq_llm.
    """

    if not model:
        return
    if model.startswith("groq:"):
        name = model.split("groq:", 1)[1].strip()
        if name:
            os.environ["LIFEOPS_GROQ_PRIMARY_MODEL"] = name
        return

    # If the user passes an unsupported format, keep defaults.
    # (Hackathon-friendly: do not crash for minor CLI mistakes.)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agents on LifeOpsEnv")
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Number of episodes to run (default: 20)")
    parser.add_argument(
        "--model",
        type=str,
        default="groq:meta-llama/llama-4-scout-17b-16e-instruct",
        help="LLM model selector (e.g., groq:llama-3.3-70b-versatile)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for scenario sampling and random policy")
    args = parser.parse_args()

    parse_model_arg(args.model)

    # Plotting is optional in headless/minimal environments. If matplotlib isn't
    # available, we still run evaluation and print metrics.
    do_plot = os.environ.get("LIFEOPS_SKIP_PLOT", "").strip() != "1"
    plt = None
    if do_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            do_plot = False

    rng = random.Random(args.seed)
    # Curriculum schedule: easy -> medium -> hard (ramp every 20 episodes).
    episode_scenarios: List[str] = []
    for ep in range(1, int(args.episodes) + 1):
        if ep <= 20:
            diff = "easy"
        elif ep <= 40:
            diff = "medium"
        else:
            diff = "hard"
        pool = scenario_ids_by_difficulty(diff)
        if not pool:
            pool = scenario_ids_by_difficulty("hard")
        episode_scenarios.append(rng.choice(pool))

    # Separate env instances for isolation/repeatability.
    env_random = LifeOpsEnv(seed=args.seed)
    env_baseline = LifeOpsEnv(seed=args.seed)
    env_llm = LifeOpsEnv(seed=args.seed)

    random_policy = make_random_policy(seed=args.seed + 1)
    baseline_policy = make_baseline_policy()
    llm_agent = LLMAgent(local_model_name=os.environ.get("LIFEOPS_LOCAL_MODEL"))
    llm_policy = make_llm_policy(llm_agent)

    rewards_random: List[float] = []
    rewards_baseline: List[float] = []
    rewards_llm: List[float] = []
    viol_random: List[Dict[str, int]] = []
    viol_baseline: List[Dict[str, int]] = []
    viol_llm: List[Dict[str, int]] = []

    for i, sid in enumerate(episode_scenarios, start=1):
        r_r, v_r, _ = run_episode(env_random, random_policy, scenario_id=sid)
        r_b, v_b, _ = run_episode(env_baseline, baseline_policy, scenario_id=sid)
        r_l, v_l, traj = run_episode(env_llm, llm_policy, scenario_id=sid, collect_trajectory=True)

        rewards_random.append(r_r)
        rewards_baseline.append(r_b)
        rewards_llm.append(r_l)
        viol_random.append(v_r)
        viol_baseline.append(v_b)
        viol_llm.append(v_l)

        # Update LLM memory after each episode.
        if traj is not None:
            llm_agent.on_episode_end(traj, total_reward=r_l)

        def most(v: Dict[str, int]) -> str:
            if not v:
                return "none"
            if max(v.values()) <= 0:
                return "none"
            return max(v.items(), key=lambda kv: kv[1])[0]

        print(
            f"Episode {i:02d}/{args.episodes} | scenario={sid} | "
            f"random={r_r:+.2f} (most={most(v_r)}) | "
            f"baseline={r_b:+.2f} (most={most(v_b)}) | "
            f"llm={r_l:+.2f} (most={most(v_l)})"
        )

    def avg(xs: List[float]) -> float:
        return (sum(xs) / float(len(xs))) if xs else 0.0

    print("\nSummary:")
    print(f"Random avg reward:   {avg(rewards_random):+.3f}")
    print(f"Baseline avg reward: {avg(rewards_baseline):+.3f}")
    print(f"LLM avg reward:      {avg(rewards_llm):+.3f}")

    if do_plot and plt is not None:
        # Plot reward curves.
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_random, label="random", alpha=0.6, color="tab:blue")
        plt.plot(rewards_llm, label="llm", alpha=0.6, color="tab:green")
        plt.plot(rewards_baseline, label="baseline", alpha=0.6, color="tab:orange")

        # Rolling averages for smoothing (requested).
        plt.plot(_moving_average(rewards_random, 10), label="random (ma10)", linewidth=2.5, color="tab:blue")
        plt.plot(_moving_average(rewards_llm, 10), label="llm (ma10)", linewidth=2.5, color="tab:green")
        plt.plot(_moving_average(rewards_baseline, 10), label="baseline (ma10)", linewidth=2.5, color="tab:orange")

        plt.title("LifeOps reward comparison")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = Path(__file__).resolve().parent / "reward_comparison.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to: {out_path}")

        # Requirement: display plot and also save it.
        plt.show()
    else:
        print("\nPlot skipped (set LIFEOPS_SKIP_PLOT=0 and install matplotlib to enable).")


if __name__ == "__main__":
    main()

