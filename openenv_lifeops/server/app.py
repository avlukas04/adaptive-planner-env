"""
FastAPI app for LifeOps OpenEnv server.
"""

from openenv.core.env_server.http_server import create_app

from ..models import LifeOpsAction, LifeOpsObservation
from .lifeops_environment import LifeOpsEnvironment

app = create_app(
    LifeOpsEnvironment,
    LifeOpsAction,
    LifeOpsObservation,
    env_name="lifeops_env",
)


def main():
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
