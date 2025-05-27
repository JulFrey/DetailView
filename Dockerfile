FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy install packages
RUN pip install numpy pandas scikit-learn laspy matplotlib requests
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

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

# Set entrypoint
ENTRYPOINT ["python", "predict.py"]

# docker build -t detailview .
# docker run --rm --gpus all -v "C:/TLS/docker/input:/input" -v "C:/TLS/docker//output:/output" detailview --prediction_data /input/circle_3_segmented.las --model_path /app/model_ft_202412171652_3