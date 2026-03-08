# Deploy LifeOps to Hugging Face Spaces

## How it works

**Hugging Face Spaces are separate Git repos** — they don't "link" to GitHub. You either:

1. **Push your code to the Space** (one-time or via GitHub Actions)
2. **Clone the Space, add your code, push back**

---

## Option A: One-time push (simplest)

### 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Owner:** your username
3. **Space name:** `lifeops`
4. **SDK:** Gradio
5. **Template:** Blank
6. Create the Space

### 2. Push your repo to the Space

```bash
# From your adaptive-planner-env directory
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/lifeops

# Authenticate: use your HF token (Settings → Access Tokens on huggingface.co)
git push --force space main
```

If prompted for credentials, use your HF username and an **Access Token** (not your password). Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Space requirements

The Space expects:
- `app.py` in the root ✅ (we have this)
- `requirements.txt` ✅ (we have this)

**Secrets:** In the Space settings, add `GROQ_API_KEY` so the LLM agent works.

---

## Option B: Auto-sync from GitHub (GitHub Actions)

Every push to `main` automatically updates the Space.

### 1. Create the Space (same as Option A, step 1)

### 2. Add GitHub Secrets

In your GitHub repo: **Settings → Secrets and variables → Actions**

- `HF_TOKEN` — your Hugging Face Access Token (with write permission)

### 3. Add repo variables (optional)

**Settings → Secrets and variables → Actions → Variables**

- `HF_USERNAME` — your HF username (defaults to repo owner)
- `HF_SPACE_NAME` — `lifeops` (default)

### 4. Push to main

The workflow in `.github/workflows/sync-to-hf-space.yml` will push to your Space.

---

## Troubleshooting

- **"Not authorized to push"** — Use an Access Token, not your password. Token needs `write` scope.
- **App fails to load** — Check the Space logs. Ensure `GROQ_API_KEY` is set in Space settings.
- **Heavy dependencies** — The full `requirements.txt` includes torch/transformers. For a lighter Space, you could use `requirements-spaces.txt` (gradio, groq, dotenv only) and rely on Groq API for the LLM.
