FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy install packages
RUN pip install numpy pandas scikit-learn laspy matplotlib requests
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy your code
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "predict.py"]

# docker build -t detailview .
# docker run --rm --gpus all -v "C:/TLS/docker/input:/input" -v "C:/TLS/docker//output:/output" a0d4411d3a8d --prediction_data input/circle_3_segmented.las