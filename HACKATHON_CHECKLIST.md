# OpenEnv Hackathon Checklist

## Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| OpenEnv 0.2.1 on HF Spaces | ⬜ | Deploy with `./scripts/deploy_openenv.sh` |
| Minimal training script (Unsloth/TRL in Colab) | ✅ | `scripts/colab_train_minimal.py` |
| 1-minute demo video on YouTube | ⬜ | Record and upload |
| Address at least one problem statement | ✅ | Statement 3.2: Personalized Tasks |

## Deployment Steps

1. **Create Space:** [huggingface.co/new-space](https://huggingface.co/new-space) → Docker SDK
2. **Deploy:** `./scripts/deploy_openenv.sh openenv-community/lifeops-env`
3. **Verify:** Run the Python snippet in `DEPLOY_OPENENV.md` Step 5
4. **Colab:** Update `LIFEOPS_ENV_URL` in the script and run

## Other Steps

- [ ] Push `openenv-integration` branch to origin
- [ ] Deploy OpenEnv env to HF Spaces
- [ ] Test Colab script end-to-end
- [ ] Record 1-minute demo video
- [ ] Submit at [cerebralvalley.ai](https://cerebralvalley.ai/e/open-env-hackathon)
