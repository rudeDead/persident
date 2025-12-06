FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install CPU pytorch
RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
