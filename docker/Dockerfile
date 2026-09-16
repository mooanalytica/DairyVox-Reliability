FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    sox \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/stage4_pipeline

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /tmp/requirements.txt

CMD ["/bin/bash"]
