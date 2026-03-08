#!/bin/sh
# Install pre-push hook to block pushing main to origin.
# Run: ./scripts/install-pre-push-hook.sh
HOOK_SRC="$(dirname "$0")/pre-push-hook"
HOOK_DST="$(git rev-parse --git-dir)/hooks/pre-push"
cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
echo "Installed pre-push hook. Pushing main to origin is now blocked."
