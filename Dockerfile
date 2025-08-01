# -----------------------------------------------------------------------------
# Phase 1: The "Builder" Stage
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir uv

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/test/cu124

# --- NEW: Download the models during the build ---
# Copy the download script into the builder
COPY download_models.py .
# Run the script to download models into a /models directory
RUN python download_models.py
# --- END NEW ---


# -----------------------------------------------------------------------------
# Phase 2: The "Final" Runtime Stage
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# No longer need HF_HOME=/runpod-volume, as models are now inside the image
# ENV HF_HOME=/runpod-volume # <-- REMOVE OR COMMENT OUT THIS LINE

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=builder /opt/venv /opt/venv

COPY ./src /app
WORKDIR /app

# --- NEW: Copy the downloaded models from the builder stage ---
COPY --from=builder /models /app/models
# --- END NEW ---

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "-u", "handler.py"]