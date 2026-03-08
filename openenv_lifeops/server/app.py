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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
