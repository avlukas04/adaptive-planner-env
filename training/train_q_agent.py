"""
Train a tabular Q-learning agent on LifeOps (MVP).

Usage (from repo root):
  python training/train_q_agent.py

Only uses: python stdlib, numpy, matplotlib.
No external RL frameworks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import random

def _ensure_repo_on_path() -> None:
    """
    Allow running this file directly via `python training/train_q_agent.py`.
    """

    try:
        import env  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from env.lifeops_env import LifeOpsEnv  # noqa: E402
from env.scenario_generator import scenario_ids_by_difficulty  # noqa: E402
from training.q_learning_agent import QLearningAgent  # noqa: E402


def moving_average(x: List[float], window: int) -> List[float]:
    """
    Simple moving average (pure Python) to keep the trainer robust.
    """

    if window <= 1:
        return list(x)
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(x):
        s += float(v)
        if i >= window:
            s -= float(x[i - window])
        denom = min(i + 1, window)
        out.append(s / float(denom))
    return out


def run_episode(env: LifeOpsEnv, agent: QLearningAgent, train: bool, *, scenario_id: Optional[str] = None) -> Tuple[float, bool]:
    """
    Runs a single episode.

    Returns:
      (total_reward, success)

    Success definition (simple + observable):
    - episode ended before max_steps (i.e., not truncated by step limit)
    """

    state: Dict[str, Any] = env.reset(scenario_id=scenario_id)
    total_reward = 0.0

    # Hard safety cap in case of a bug.
    safety_cap = int(state.get("max_steps", 50)) + 50
    steps = 0

    done = False
    while not done and steps < safety_cap:
        steps += 1
        valid_actions = env.valid_actions()
        if not valid_actions:
            # Should not happen; treat as done with current reward.
            break

        action = agent.select_action(state, valid_actions)

        try:
            next_state, reward, done, info = env.step(action)
        except Exception:
            # Safety: never crash training loop on an unexpected env error.
            # Penalize and terminate the episode.
            total_reward -= 10.0
            break

        total_reward += float(reward)

        if train:
            next_valid = env.valid_actions() if not done else []
            agent.update(state, action, float(reward), next_state, next_valid, bool(done))

        state = next_state

    # If we hit the safety cap, consider it not successful.
    truncated = bool(state.get("step_count", 0) >= state.get("max_steps", 0)) or (steps >= safety_cap)
    success = not truncated
    return total_reward, success


def main() -> None:
    episodes = 200
    eval_episodes = 20
    ma_window = 20

    env = LifeOpsEnv(seed=42)
    agent = QLearningAgent(
        epsilon=0.25,
        learning_rate=0.15,
        discount_factor=0.95,
        seed=123,
    )

    episode_rewards: List[float] = []
    episode_success: List[bool] = []

    for ep in range(1, episodes + 1):
        # Curriculum: start easy, ramp every 20 episodes.
        if ep <= 20:
            difficulty = "easy"
        elif ep <= 40:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Light epsilon decay helps converge in a small tabular setting.
        agent.epsilon = max(0.05, agent.epsilon * 0.995)

        # Pick a scenario id from the current difficulty bucket.
        pool = scenario_ids_by_difficulty(difficulty)
        scenario_id = random.choice(pool) if pool else None

        r, success = run_episode(env, agent, train=True, scenario_id=scenario_id)
        episode_rewards.append(float(r))
        episode_success.append(bool(success))

        if ep % 20 == 0:
            last20 = episode_rewards[-20:]
            ma = (sum(last20) / float(len(last20))) if last20 else 0.0
            last20s = episode_success[-20:]
            sr = (sum(1.0 for s in last20s if s) / float(len(last20s))) if last20s else 0.0
            print(
                f"Episode {ep:4d}/{episodes} | diff={difficulty:6s} | "
                f"last20 avg reward={ma:8.3f} | last20 success={sr:5.2%} | eps={agent.epsilon:.3f}"
            )

    ma = moving_average(episode_rewards, ma_window)

    # Evaluation (greedy policy)
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    eval_rewards: List[float] = []
    eval_success: List[bool] = []
    hard_pool = scenario_ids_by_difficulty("hard")
    for _ in range(eval_episodes):
        sid = random.choice(hard_pool) if hard_pool else None
        r, success = run_episode(env, agent, train=False, scenario_id=sid)
        eval_rewards.append(float(r))
        eval_success.append(bool(success))
    agent.epsilon = old_eps

    print("\nEvaluation (20 episodes, greedy):")
    avg_reward = (sum(eval_rewards) / float(len(eval_rewards))) if eval_rewards else 0.0
    success_rate = (sum(1.0 for s in eval_success if s) / float(len(eval_success))) if eval_success else 0.0
    print(f"  average reward: {avg_reward:.3f}")
    print(f"  success rate:  {success_rate:.2%}")

    # Visualization (matplotlib) in a subprocess for robustness:
    # - avoids crashing the training loop if numpy/matplotlib are misinstalled
    # - still produces the required plot when dependencies are healthy
    out_dir = Path(__file__).resolve().parent
    log_path = out_dir / "training_log.json"
    plot_path = out_dir / "training_rewards.png"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_rewards": episode_rewards,
                "moving_average_rewards": ma,
                "ma_window": ma_window,
                "episodes": episodes,
            },
            f,
        )

    if os.environ.get("LIFEOPS_SKIP_PLOT", "").strip() == "1":
        print(f"\nSkipping plot (LIFEOPS_SKIP_PLOT=1). Data saved to: {log_path}")
        return

    code = r"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = Path(r"{log_path}")
plot_path = Path(r"{plot_path}")
data = json.loads(log_path.read_text(encoding="utf-8"))
rewards = data["episode_rewards"]
ma = data["moving_average_rewards"]
ma_window = data.get("ma_window", 20)

plt.figure(figsize=(10, 5))
plt.plot(rewards, label="episode reward", alpha=0.6)
plt.plot(ma, label=f"{{ma_window}}-episode moving avg", linewidth=2.5)
plt.title("LifeOps Q-learning training rewards")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
print(str(plot_path))
""".format(log_path=str(log_path), plot_path=str(plot_path))

    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if proc.returncode == 0:
            saved_to = (proc.stdout or "").strip() or str(plot_path)
            print(f"\nSaved plot to: {saved_to}")
        else:
            print(f"\nPlot subprocess failed (exit={proc.returncode}). Training data saved to: {log_path}")
            if proc.stderr:
                print(proc.stderr.strip())
    except Exception as e:
        print(f"\nPlotting failed ({e}). Training data saved to: {log_path}")


if __name__ == "__main__":
    main()

