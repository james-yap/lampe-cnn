FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/lampe-venv \
    VIRTUAL_ENV=/opt/lampe-venv \
    PATH="/opt/lampe-venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    LAMPE_DATASET_DIR=/data/lampe/dataset \
    LAMPE_MODEL_PATH=/data/lampe/model_v2.pth

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 lampe \
    && mkdir -p /workspace/lampe /opt/lampe-venv /data/lampe \
    && chown -R lampe:lampe /workspace /opt/lampe-venv /data/lampe

WORKDIR /workspace/lampe

COPY --chown=lampe:lampe pyproject.toml uv.lock README.md ./

USER lampe

RUN uv sync --frozen --no-install-project

COPY --chown=lampe:lampe . .

RUN uv sync --frozen \
    && uv pip install -e .

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"]
