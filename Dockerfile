FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# are those really needed, do we really compile anything here?
# if we compile we should use a 2-stage build and NOT have those build dependencies in the runtime container
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy install packages
# I hope this will save some space
RUN pip3 install --no-cache-dir numpy pandas scikit-learn laspy matplotlib requests tqdm && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir lazrs[all]
#RUN pip3 install fastapi uvicorn

# Copy your code
COPY *.py /app/
COPY model_ft_202412171652_3 /app/
COPY lookup.csv /app/

# Set environment variable for torch cache
ENV TORCH_HOME=/app/torch_cache

# Create cache directory
RUN mkdir -p /app/torch_cache/hub/checkpoints

# Download densenet201 weights
RUN wget -O /app/torch_cache/hub/checkpoints/densenet201-c1103571.pth \
    https://download.pytorch.org/models/densenet201-c1103571.pth

# what is this doing?
RUN python3 -c "import torch; print(torch.cuda.is_available()); import time; time.sleep(2.5)"

# Set entrypoint
ENTRYPOINT ["python", "predict.py"]
#ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t detailview .
# docker run --rm --gpus all -v "C:/TLS/docker/input:/input" -v "C:/TLS/docker//output:/output" detailview --prediction_data /input/circle_3_segmented.las --model_path /app/model_ft_202412171652_3
