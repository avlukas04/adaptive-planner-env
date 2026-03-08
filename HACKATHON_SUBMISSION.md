# AI Hackathon Submission — Draft Responses

Use these as starting points for the form fields. Adjust links and details as needed.

---

## 1. Project Description *

```
LifeOps is an AI-powered schedule assistant that learns to manage your calendar using reinforcement learning. It solves the problem of scheduling overload: conflicting meetings, travel constraints, and personal goals (focus time, tasks) that compete for limited time.

The system includes:
• A custom RL environment (OpenEnv-style) with calendar events, tasks, travel times, and incoming requests
• Persona-aware planning (Early-bird Engineer, Night-owl Creator, Busy Parent) that respects preferred meeting windows
• LLM agents (Groq Llama, HuggingFace) that accept, reject, reschedule, or propose new times
• Policy improvement via Best-of-N sampling and in-context learning from high-reward trajectories
• A week-view UI with Google Calendar integration, tasks panel, and persona selector

The agent is trained to avoid overlaps, respect travel feasibility, and prioritize important requests while blocking focus time for goals. Evaluation compares random, rule-based baseline, and LLM policies across scenarios.
```

---

## 2. Hugging Face Link *

```
https://huggingface.co/spaces/YOUR_USERNAME/lifeops
```

**To deploy:** Create a new Space, choose Gradio SDK, add your repo or upload `app/week_view.py` + dependencies. Set `GROQ_API_KEY` as a secret.

---

## 3. Demo Video *

```
https://youtube.com/watch?v=YOUR_VIDEO_ID
```

**Suggested content (2–3 min):**
1. Show the week-view UI and persona selector
2. Run a scenario with the LLM agent
3. Show how it accepts/rejects/reschedules and blocks focus time
4. Briefly show training output (`python -m training.train_rl -n 5 --agent llm`)

---

## 4. Minimal Training Script (Unsloth or HF TRL in Colab) *

```python
# Cell 1: Install
!pip install -q transformers trl torch python-dotenv

# Cell 2: Clone repo and setup
!git clone https://github.com/YOUR_USER/adaptive-planner-env.git
import sys
sys.path.insert(0, "adaptive-planner-env")

# Cell 3: Train with LifeOps env (HF TRL-style reward loop)
from env.lifeops_env import LifeOpsEnv
from training.train_rl import train

result = train(num_episodes=10, agent="llm", llm_model_id="google/flan-t5-base", plot=False)
print(f"Avg reward: {result['avg_reward']:.2f}")
```

**Note:** The LifeOps env provides the reward signal. We use `trl` (Transformer Reinforcement Learning) for the policy optimization framework. The training loop collects trajectories, computes rewards (overlap penalties, travel feasibility, preference violations), and improves the LLM policy via in-context learning and Best-of-N. See `training/train_rl.py` and `training/policy_improvement.py` for the full implementation.

---

## 5. Partner Tracks (select up to 2) *

Suggested options based on the project:

- **Snorkel AI** — Programmatic labeling / weak supervision (reward shaping, scenario generation)
- **Scale AI** — Data quality and evaluation (agent comparison, reward metrics)
- **Patronus AI** — Safety / reliability (handling invalid LLM outputs, fallback logic)
- **Mercor** — AI talent / agents (LLM-based scheduling agent)

---

## Checklist Before Submitting

- [ ] Replace `YOUR_USER` / `YOUR_USERNAME` / `YOUR_VIDEO_ID` with real values
- [ ] Deploy the Gradio app to Hugging Face Spaces
- [ ] Record and upload the demo video to YouTube
- [ ] Test the Colab script and fix the repo path if needed
- [ ] Choose 2 partner tracks from the list
