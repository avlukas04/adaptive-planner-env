# Deploy LifeOps OpenEnv to HF Spaces

To meet the hackathon requirement: **OpenEnv (stable 0.2.1) deployed on HF Spaces**.

## Prerequisites

- Hugging Face account with [token](https://huggingface.co/settings/tokens) (write access)
- Git with `openenv-integration` branch pushed to origin

## Step 1: Create the HF Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** `lifeops-env` (or `YOUR_ORG-lifeops-env`)
3. **SDK:** Select **Docker**
4. **Visibility:** Public
5. Click **Create Space**

## Step 2: Deploy

**Option A: Use the deploy script (recommended)**

```bash
# From repo root, on openenv-integration branch
chmod +x scripts/deploy_openenv.sh
./scripts/deploy_openenv.sh openenv-community/lifeops-env
```

When prompted for password, use your HF token.

**Option B: Manual push**

```bash
# Add the Space as a remote
git remote add lifeops-env https://huggingface.co/spaces/openenv-community/lifeops-env

# Push (replace org/space with yours)
git push lifeops-env openenv-integration:main
```

Then in the Space's Files tab, ensure `README.md` has `sdk: docker` and `app_port: 7860` in the YAML frontmatter. Copy from `README.lifeops-env` if needed.

## Step 3: Wait for build

HF will build the Docker image. This can take 5–10 minutes. Check the Space's **Logs** tab for progress.

## Step 4: Space URL

After deployment:

```
https://openenv-community-lifeops-env.hf.space
```

(Replace `openenv-community` and `lifeops-env` with your org and space name. Use `-` instead of `/` in the URL.)

## Step 5: Verify

```python
from openenv_lifeops import LifeOpsEnv, LifeOpsAction

client = LifeOpsEnv(base_url="https://openenv-community-lifeops-env.hf.space")
result = client.reset()
print(result.observation.observation["scenario_id"])
client.close()
```

## Step 6: Use in Colab

In `scripts/colab_train_minimal.py`, set:

```python
LIFEOPS_ENV_URL = "https://YOUR_ORG-YOUR_SPACE.hf.space"
```

Then run Cell 3a to connect to the remote env.

## Troubleshooting

- **Build fails:** Check Logs for errors. Ensure `env/` and `openenv_lifeops/` are in the repo.
- **Connection refused:** Wait for the build to finish. HF Spaces can take a few minutes to start.
- **Port issues:** The Dockerfile uses port 7860 (HF default). `README.lifeops-env` sets `app_port: 7860`.
