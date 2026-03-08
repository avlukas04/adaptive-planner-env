# Branch Workflow — Keep Changes on Branch Only

To keep your work on `andreas-branch` and avoid pushing to `main`:

## Optional: Install pre-push hook

Blocks `git push origin main` by default:

```bash
./scripts/install-pre-push-hook.sh
```

## Push only your branch

```bash
# Push your branch to GitHub (origin)
git push origin andreas-branch
```

## Do NOT push main

```bash
# Avoid: this would push main to origin
# git push origin main
```

## Push to Hugging Face Space

To deploy your branch to the HF Space (updates the Space with your branch):

```bash
git push space andreas-branch:main
```

This pushes your local `andreas-branch` to the Space’s `main` branch.

## Quick checklist

- [ ] Work and commit on `andreas-branch` (check with `git branch`)
- [ ] Push branch: `git push origin andreas-branch`
- [ ] Push to Space: `git push space andreas-branch:main`
- [ ] Do not run `git push origin main`
