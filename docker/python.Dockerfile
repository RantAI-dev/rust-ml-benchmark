# Use an official lightweight Python image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# For now, we keep this minimal. Add packages like 'build-essential' if pip fails on certain libraries.
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python packages
# Note: This Dockerfile is for CPU-based tasks. For GPU support, a different
# base image (e.g., nvidia/cuda) and PyTorch version would be required.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# By default, this container will execute a bash shell.
# We will specify the command to run when we execute `docker run`.
CMD ["/bin/bash"]
