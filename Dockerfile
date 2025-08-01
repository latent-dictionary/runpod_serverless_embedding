# ---- Builder Stage ----
# Use a full SDK image to build our environment
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install uv, our fast package installer
RUN pip install uv

# Create a virtual environment. This is a best practice to keep dependencies isolated.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first to leverage Docker's cache
COPY requirements.txt .
# Install all Python dependencies, including the specific torch version, in one step
RUN uv pip install --no-cache -r requirements.txt

# ---- Final Runtime Stage ----
# Use the lighter runtime image for the final product
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set the Hugging Face cache to a persistent volume (Your excellent optimization)
ENV HF_HOME=/runpod-volume

# Install necessary runtime dependencies (libgl1 might be needed by a library)
RUN apt-get update && apt-get install -y \
    python3.11 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY ./src /app
WORKDIR /app

# Set the PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the command to run your application
# The -u flag ensures that logs are sent straight to stdout without buffering
CMD ["python", "-u", "handler.py"]