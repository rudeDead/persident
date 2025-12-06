# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by RDKit drawing + Cairo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    libfreetype6 \
    libfontconfig1 \
    libeigen3-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pytorch CPU wheels
RUN pip install --upgrade pip && \
    pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements first for caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create outputs/ if not exists
RUN mkdir -p outputs


EXPOSE 8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--preload"]
