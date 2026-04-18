FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

# Controls which fastembed variant gets installed:
#   cpu -> fastembed (CPU-only onnxruntime, works everywhere)
#   gpu -> fastembed-gpu (requires CUDA; enabled via docker-compose.gpu.yml)
ARG FASTEMBED_EXTRA=cpu

COPY pyproject.toml ./
RUN pip install --break-system-packages --no-cache-dir uv \
    && uv pip install --system --break-system-packages ".[${FASTEMBED_EXTRA}]"

COPY app ./app
COPY apps_sdk ./apps_sdk
RUN mkdir -p /app/raw_data
COPY README.md ./

ENV LISTINGS_RAW_DATA_DIR=/app/raw_data \
    LISTINGS_DB_PATH=/data/listings.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
