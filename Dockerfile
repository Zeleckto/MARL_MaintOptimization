# Dockerfile
# ==========
# Manufacturing MARL environment.
# Builds a reproducible container with all dependencies.
# RTX 3060 compatible — uses CUDA 12.4 base image.

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Pygame display dependencies
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install PyG dependencies
RUN pip install torch-geometric && \
    pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Install remaining requirements
RUN pip install pettingzoo gymnasium ortools pygame pyyaml \
    tensorboard pytest scipy matplotlib

# Copy project code
COPY . .

# Create directories for outputs
RUN mkdir -p runs checkpoints benchmarks/results

# Default command: run tests to verify installation
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
