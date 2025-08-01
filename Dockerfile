# -----------------------------------------------------------------------------
# Phase 1: The "Builder" Stage
# We use a larger 'devel' image here that has all the build tools we need.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Set the Hugging Face home to a directory we can cache during builds if needed.
# This ENV is not strictly needed in the builder but is good practice.
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies and Python.
# Using DEBIAN_FRONTEND=noninteractive prevents prompts during installation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Link python3.11 to python for convenience
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install uv, the fast package installer
RUN pip install uv

# Create a virtual environment in a standard location. This is key for isolation.
RUN python -m venv /opt/venv

# Activate the venv for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy just the requirements file first to leverage Docker's layer cache.
COPY requirements.txt .

# Install dependencies from requirements.txt using uv.
# We exclude torch here to install it separately and specifically.
RUN uv pip install --no-cache -r requirements.txt --system

# NOW, install the specific torch version that works for you.
# This preserves the exact command from your working file.
RUN pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/test/cu124 --no-cache-dir


# -----------------------------------------------------------------------------
# Phase 2: The "Final" Runtime Stage
# We use the much smaller 'runtime' image for our final product.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set the RunPod persistent volume for model caching (your excellent optimization)
ENV HF_HOME=/runpod-volume

# Install only the essential runtime system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy the entire virtual environment with all its installed packages from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code into a dedicated 'app' directory.
COPY ./src /app
WORKDIR /app

# Ensure the PATH includes our virtual environment's executables.
ENV PATH="/opt/venv/bin:$PATH"

# Set the command to run your application handler.
# The -u flag is important for unbuffered logging in containers.
CMD ["python", "-u", "handler.py"]