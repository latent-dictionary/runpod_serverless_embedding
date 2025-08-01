# -----------------------------------------------------------------------------
# Phase 1: The "Builder" Stage
# This stage only prepares the Python environment. It does NOT download models.
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
# Note: huggingface-hub is now a dependency for the final stage, 
# so it MUST be in requirements.txt.
RUN uv pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/test/cu124


# -----------------------------------------------------------------------------
# Phase 2: The "Final" Runtime Stage
# This stage now downloads the models directly.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy the prepared Python environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source AND the download script
COPY ./src /app
COPY download_models.py /app/download_models.py
WORKDIR /app

# ---- THIS IS THE KEY CHANGE ----
# Run the download script inside the final stage.
# This bakes the models directly into the final image layers.
RUN python download_models.py

# Set the command to run your application handler
CMD ["python", "-u", "handler.py"]