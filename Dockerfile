# LifeOps OpenEnv environment for HF Spaces
# Use python base (avoids openenv-base which can hit HF DNS issues during build)
FROM python:3.10-slim

WORKDIR /app

# HF Spaces runs containers as UID 1000 - create user to avoid permission issues
RUN useradd -m -u 1000 user

# Install openenv-core and deps (PyPI only, no HF during build)
RUN pip install --no-cache-dir openenv-core==0.2.1 fastapi uvicorn

# Copy env and openenv_lifeops (--chown for HF Spaces compatibility)
COPY --chown=user env /app/env
COPY --chown=user openenv_lifeops /app/openenv_lifeops
COPY --chown=user scripts/start_server.sh /app/scripts/start_server.sh
RUN chmod +x /app/scripts/start_server.sh

ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

USER user

# HEALTHCHECK: /health must be served by app (openenv_lifeops/server/app.py)
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["/app/scripts/start_server.sh"]
