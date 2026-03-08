#!/bin/sh
# Startup wrapper - prints before any Python imports to debug HF Spaces
set -e
echo "[LifeOps] Container CMD started" 1>&2
echo "[LifeOps] Starting uvicorn..." 1>&2
exec python -u -m uvicorn openenv_lifeops.server.app:app --host 0.0.0.0 --port 7860
