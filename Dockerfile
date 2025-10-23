# Build stage (optional, if you need to compile wheels)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS builder

WORKDIR /app

# Install python and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev wget && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip3 install --no-cache-dir numpy pandas scikit-learn laspy requests tqdm lazrs[all]

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy app code and required assets only
COPY *.py /app/
COPY lookup.csv /app/

# Create directories for model weights
RUN mkdir -p /app/torch_cache/hub/checkpoints

# Download model weights
RUN wget -O /app/torch_cache/hub/checkpoints/densenet201-c1103571.pth \
    https://download.pytorch.org/models/densenet201-c1103571.pth

# ---

# Final stage: minimal runtime
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install minimal runtime dependencies only
RUN apt-get update && \
    apt-get install -y libexpat1 && \
    rm -rf /var/lib/apt/lists/*

# Copy complete Python installation from builder
COPY --from=builder /usr/bin/python3* /usr/bin/
COPY --from=builder /usr/lib/python3.10 /usr/lib/python3.10
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app /app

# ENV and cache setup
ENV TORCH_HOME=/app/torch_cache
RUN mkdir -p /app/torch_cache/hub/checkpoints

# Create input/output directories
RUN mkdir -p /out && chmod -R 777 /out && \
    mkdir -p /in && chmod -R 777 /in

ENTRYPOINT ["python3", "predict.py"]