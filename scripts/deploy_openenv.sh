#!/bin/sh
# Deploy LifeOps OpenEnv env to HF Spaces
# Usage: ./scripts/deploy_openenv.sh [org/space-name]
# Example: ./scripts/deploy_openenv.sh avlukas/lifeops-openenv
#
# IMPORTANT: Use a Space created with Docker SDK. Repushing to a Space
# that was created with Gradio/Streamlit often causes "stuck on Starting".

set -e

SPACE_ID="${1:-avlukas/lifeops-openenv}"
REMOTE_NAME="lifeops-env"

echo "Deploying to $SPACE_ID..."

# Add or update remote to use the passed SPACE_ID
REMOTE_URL="https://huggingface.co/spaces/$SPACE_ID"
if git remote get-url "$REMOTE_NAME" 2>/dev/null; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

# Temp branch with Docker README for the Space
git checkout -B deploy-lifeops-env openenv-integration
cp README.lifeops-env README.md
git add README.md
git commit -m "Docker README for HF Space" || true

# Push (use HF token as password when prompted)
# --force: overwrite HF's initial Space template with our deployment
git push --force "$REMOTE_NAME" deploy-lifeops-env:main

# Restore
git checkout openenv-integration
git branch -D deploy-lifeops-env

echo ""
echo "Done. Space: https://huggingface.co/spaces/$SPACE_ID"
echo "Env URL: https://${SPACE_ID//\//-}.hf.space"
