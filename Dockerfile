# -----------------------------------------------------------------------------
# Phase 1: The "Builder" Stage
# Use a full 'devel' image with build tools.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python, and crucially, the venv module.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \ 
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Link python3.11 and pip3 to default 'python' and 'pip' commands
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install uv, the fast package installer
RUN pip install --no-cache-dir uv

# Create the virtual environment. This will now succeed.
RUN python -m venv /opt/venv

# Activate the venv for all subsequent RUN commands in this stage
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file first to leverage Docker's layer cache
COPY requirements.txt .

# Install dependencies from requirements.txt using uv
# As discussed, torch should not be in this file.
RUN uv pip install --no-cache-dir -r requirements.txt

# Install the specific torch version that you confirmed works
RUN pip install --no-cache-dir torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/test/cu124


# -----------------------------------------------------------------------------
# Phase 2: The "Final" Runtime Stage
# Start fresh from the smaller 'runtime' image.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set the RunPod persistent volume for model caching
ENV HF_HOME=/runpod-volume

# Install only the absolute essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Link python for consistency
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy the entire virtual environment with all its installed packages from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code into a dedicated 'app' directory
COPY ./src /app
COPY download_models.py /app/download_models.py

WORKDIR /app
RUN python download_models.py


# Set the PATH to use the virtual environment's executables in the final container
ENV PATH="/opt/venv/bin:$PATH"

# Set the command to run your application handler with unbuffered output
CMD ["python", "-u", "handler.py"]