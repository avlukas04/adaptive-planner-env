#!/bin/sh
# Deploy LifeOps OpenEnv env to HF Spaces
# Usage: ./scripts/deploy_openenv.sh [org/space-name]
# Example: ./scripts/deploy_openenv.sh openenv-community/lifeops-env

set -e

SPACE_ID="${1:-openenv-community/lifeops-env}"
REMOTE_NAME="lifeops-env"

echo "Deploying to $SPACE_ID..."

# Add remote if not exists
if ! git remote get-url "$REMOTE_NAME" 2>/dev/null; then
  git remote add "$REMOTE_NAME" "https://huggingface.co/spaces/$SPACE_ID"
fi

# Temp branch with Docker README for the Space
git checkout -B deploy-lifeops-env openenv-integration
cp README.lifeops-env README.md
git add README.md
git commit -m "Docker README for HF Space" || true

# Push (use HF token as password when prompted)
git push "$REMOTE_NAME" deploy-lifeops-env:main

# Restore
git checkout openenv-integration
git branch -D deploy-lifeops-env

echo ""
echo "Done. Space: https://huggingface.co/spaces/$SPACE_ID"
echo "Env URL: https://${SPACE_ID//\//-}.hf.space"
