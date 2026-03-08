# adaptive-planner-env

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python training/train_rl.py -n 10 --agent random --no-plot
python training/train_rl.py -n 10 --agent baseline --no-plot
python training/train_rl.py -n 10 --agent llm --no-plot
```

For `--agent llm`, `requirements.txt` includes `transformers` and `torch`. Use `--model google/flan-t5-base` to force a specific model.

### Groq API (Llama 4, Llama 3.3, etc.)

Use Groq-hosted models for fast inference without local GPU:

1. Create an API key at [console.groq.com](https://console.groq.com)
2. Add to `.env`: `GROQ_API_KEY=your_key_here`
3. Run with `--model "groq:llama-3.3-70b-versatile"` or `--model "groq:meta-llama/llama-4-scout-17b-16e-instruct"`

The API key is account-level; you choose the model per request. No need to "attach" a model to the key.

If you see HuggingFace cache permission errors, use a project-local cache:
```bash
HF_HOME=.cache/huggingface python training/train_rl.py -n 10 --agent llm --no-plot
```
Or run `./scripts/run_llm_training.sh 10`.

### Policy improvement (RL-style methods)

Improve LLM performance using reward-based techniques (no gradients):

| Method | Description | Usage |
|--------|-------------|-------|
| `vanilla` | Greedy LLM (default) | `--method vanilla` |
| `best-of-n` | Sample N actions per step, simulate each, pick the one with highest immediate reward | `--method best-of-n --best-of-n 5` |
| `in-context` | Maintain replay buffer of best trajectories; add few-shot examples to the prompt | `--method in-context --in-context-size 5` |

Example:
```bash
python -m training.train_rl -n 20 --agent llm --model "groq:llama-3.3-70b-versatile" --method best-of-n --best-of-n 5
python -m training.train_rl -n 50 --agent llm --model "groq:llama-3.3-70b-versatile" --method in-context
```

## Evaluation

Compare agent policies (random, baseline, LLM):

```bash
python evaluation/evaluate_agents.py -n 20
```

Options: `-n 20` (episodes per agent), `--agents random baseline llm`, `--seed 42`, `--save-plot rewards.png`, `--no-plot` (skip display).

## Gradio Demo

**Week view UI** (recommended) — week calendar, events, tasks, persona selector:

```bash
PYTHONPATH=. python -m app.week_view
```

**Simple demo** — scenario + agent simulation:

```bash
python app/demo.py
```

### Google Calendar integration

To sync with Google Calendar:

1. Enable the Calendar API in [Google Cloud Console](https://console.cloud.google.com)
2. Create OAuth credentials (Desktop app), download as `credentials.json` in the project root
3. Install: `pip install google-api-python-client google-auth-oauthlib`
4. Run the week view; on first load it will open a browser to authorize
