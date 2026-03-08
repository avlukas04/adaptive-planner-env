# LifeOps OpenEnv environment for HF Spaces
# Use python base (avoids openenv-base which can hit HF DNS issues during build)
FROM python:3.10-slim

WORKDIR /app

# Install openenv-core and deps (PyPI only, no HF during build)
RUN pip install --no-cache-dir openenv-core==0.2.1 fastapi uvicorn

# Copy env and openenv_lifeops
COPY env /app/env
COPY openenv_lifeops /app/openenv_lifeops

ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "openenv_lifeops.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
