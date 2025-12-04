FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libpq-dev curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
RUN apt-get update && apt-get install -y libpq5 curl libgomp1 && rm -rf /var/lib/apt/lists/*
RUN addgroup --system aiuser && adduser --system --ingroup aiuser --uid 1000 aiuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY . /app
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embeddings /app/models /app/logs
RUN chown -R aiuser:aiuser /app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
# Ensure entrypoint script is executable
RUN chmod +x /app/validate-and-start.sh || true
USER aiuser
CMD ["sh","/app/validate-and-start.sh"]

# GPU variant notes:
# To enable GPU support, replace the base images with appropriate CUDA-enabled images and
# install PyTorch with CUDA support. Also run `docker-compose up --build --force-recreate` with
# the service set to use the host GPU (e.g. `deploy.resources.reservations.devices` or `--gpus all`).
