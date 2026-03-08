# Deploy LifeOps OpenEnv to HF Spaces

To meet the hackathon requirement: **OpenEnv (stable 0.2.1) deployed on HF Spaces**.

**Scope:** This applies to the **entire project**, not just Colab. The OpenEnv environment must be deployed on HF Spaces, and any training (Colab, local, etc.) can connect to it.

## 1. Create a new HF Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Name: `lifeops-env` (or `YOUR_ORG-lifeops-env`)
4. **SDK: Docker**
5. Visibility: Public

## 2. Push the OpenEnv environment

Push the repo contents to the Space. The Space needs:

- `Dockerfile` (use `Dockerfile.openenv` as the Dockerfile)
- `env/` folder
- `openenv_lifeops/` folder

**Option A: From repo root**

```bash
# Copy Dockerfile.openenv to Dockerfile for the Space
cp Dockerfile.openenv Dockerfile

# Add the Space as a remote (if not already)
git remote add lifeops-env https://huggingface.co/spaces/YOUR_ORG/lifeops-env

# Push (use HF token when prompted)
git push lifeops-env main
```

**Option B: Manual upload**

1. In the Space, go to Files
2. Upload `Dockerfile.openenv` and rename to `Dockerfile`
3. Upload the `env/` folder
4. Upload the `openenv_lifeops/` folder

## 3. Space URL

After deployment, the Space URL will be:

```
https://YOUR_ORG-lifeops-env.hf.space
```

Use this in the Colab script as `LIFEOPS_ENV_URL`.

## 4. Verify

```python
from openenv_lifeops import LifeOpsEnv, LifeOpsAction

client = LifeOpsEnv(base_url="https://YOUR_ORG-lifeops-env.hf.space")
result = client.reset()
print(result.observation.observation["scenario_id"])
client.close()
```
