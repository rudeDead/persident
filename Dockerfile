FROM python:3.10-slim

WORKDIR /app
COPY . /app

# install system deps for RDKit drawing + compiling libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install CPU pytorch
RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
