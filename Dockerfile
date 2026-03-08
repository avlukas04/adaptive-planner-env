# LifeOps OpenEnv environment for HF Spaces
# HF Spaces uses port 7860 by default
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy env and openenv_lifeops
COPY env /app/env
COPY openenv_lifeops /app/openenv_lifeops

ENV PYTHONPATH="/app:$PYTHONPATH"

# Install openenv-core and deps
RUN pip install --no-cache-dir openenv-core==0.2.1 fastapi uvicorn

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "openenv_lifeops.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
