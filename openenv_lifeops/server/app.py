"""
FastAPI app for LifeOps OpenEnv server.
"""
import sys
print("[LifeOps] Loading app...", flush=True)
sys.stdout.flush()

from openenv.core.env_server.http_server import create_app
print("[LifeOps] create_app imported", flush=True)

from ..models import LifeOpsAction, LifeOpsObservation
from .lifeops_environment import LifeOpsEnvironment
print("[LifeOps] Models and env imported", flush=True)

app = create_app(
    LifeOpsEnvironment,
    LifeOpsAction,
    LifeOpsObservation,
    env_name="lifeops_env",
)
print("[LifeOps] App created, ready for uvicorn", flush=True)


@app.get("/health")
def health():
    """HF Spaces / Docker HEALTHCHECK endpoint."""
    return {"status": "ok"}


def main():
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
