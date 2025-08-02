# Start from an official lightweight Rust image based on Debian Bullseye
FROM rust:1.79-slim-bullseye

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies required for building common crates
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    wget \
    unzip \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Install LibTorch (CPU version for tch-rs) ---
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip -O libtorch.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip

# --- Install ONNX Runtime (for ort) ---
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz -O onnxruntime.tgz && \
    tar -zxvf onnxruntime.tgz && \
    rm onnxruntime.tgz

# --- Set Environment Variables ---
# Define paths for the downloaded libraries
ENV LIBTORCH_DIR=/app/libtorch
ENV ONNXRUNTIME_DIR=/app/onnxruntime-linux-x64-1.13.1

# Set the library path for both LibTorch and ONNX Runtime.
# This single instruction avoids overwriting and resolves the undefined variable warning.
ENV LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:${ONNXRUNTIME_DIR}/lib

# Copy the Rust project files into the container
# We copy only the rust projects to optimize Docker layer caching
COPY ./experiments/rust/ /app/

# By default, this container will execute a bash shell.
CMD ["/bin/bash"]
