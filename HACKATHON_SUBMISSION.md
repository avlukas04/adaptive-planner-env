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

See `scripts/colab_train_minimal.py` for the full script. Copy each Cell block into a Colab cell.

**Cell 1 – Install:** `!pip install -q transformers trl torch datasets accelerate python-dotenv`

**Cell 2 – Clone:** `!git clone -q https://github.com/avlukas04/adaptive-planner-env.git` then `sys.path.insert(0, "adaptive-planner-env")`

**Cell 3 – TRL SFT:** Collects high-reward trajectories from the LifeOps env, builds a dataset of (prompt, completion) pairs, and fine-tunes with `trl.SFTTrainer` on `google/flan-t5-small` (Colab-friendly). Optional: use Unsloth for faster LoRA training.

**Cell 4 (alternative):** Run `train(num_episodes=10, agent="baseline", plot=False)` for env-only training (Best-of-N, in-context learning).

**Note:** The LifeOps env provides the reward signal. TRL SFT fine-tunes on high-reward trajectories; the built-in loop uses Best-of-N and in-context learning. See `training/train_rl.py` and `training/policy_improvement.py`.

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
