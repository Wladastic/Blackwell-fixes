# Blackwell-fixes
A general Repo for guides and notes I take when trying to get Cuda and other Stuff running on Blackwell

I am running a Gigabyte GeForce RTX 5060 Ti Eagle OC 16GB as GPU which is based on Blackwell with Compute Capability arch version "12.0".

Generally it runs pretty well.

Current Cuda Version working: 13.0


Issues:
Docker Containers do not always recognize Cuda.


## Chatterbox-tts-api

I am having issues with getting the docker compose to run of:
https://github.com/travisvn/chatterbox-tts-api


The issue: Once it starts, it downloads 6 large model files, then fails as it only checks then if torch can even find cuda devices..
In this case I ran:
docker compose -f docker/docker-compose.gpu.yml --profile frontend up --build

```
chatterbox-tts-api-gpu   | Starting Chatterbox TTS API server...
chatterbox-tts-api-gpu   | Server will run on http://0.0.0.0:4123
chatterbox-tts-api-gpu   | API documentation available at http://0.0.0.0:4123/docs
chatterbox-tts-api-gpu   | INFO:     Started server process [1]
chatterbox-tts-api-gpu   | INFO:     Waiting for application startup.
chatterbox-tts-api-gpu   | 
chatterbox-tts-api-gpu   |   ____ _           _   _            _               
chatterbox-tts-api-gpu   |  / ___| |__   __ _| |_| |_ ___ _ __| |__   _____  __
chatterbox-tts-api-gpu   | | |   | '_ \ / _` | __| __/ _ \ '__| '_ \ / _ \ \/ /
chatterbox-tts-api-gpu   | | |___| | | | (_| | |_| ||  __/ |  | |_) | (_) >  < 
chatterbox-tts-api-gpu   |  \____|_| |_|\__,_|\__|\__\___|_|  |_.__/ \___/_/\_\
chatterbox-tts-api-gpu   |                                                     
chatterbox-tts-api-gpu   | 
chatterbox-tts-api-gpu   | Initializing voice library...
chatterbox-tts-api-gpu   | Using system default voice
chatterbox-tts-api-gpu   | Starting long text background processor...
chatterbox-tts-api-gpu   | Long text background processor started
chatterbox-tts-api-gpu   | Initializing Chatterbox TTS model...
chatterbox-tts-api-gpu   | Device: cuda
chatterbox-tts-api-gpu   | Voice sample: /app/voice-sample.mp3
chatterbox-tts-api-gpu   | Model cache: /cache
chatterbox-tts-api-gpu   | Loading Chatterbox Multilingual TTS model...
chatterbox-tts-api-gpu   | INFO:     Application startup complete.
chatterbox-tts-api-gpu   | INFO:     Uvicorn running on http://0.0.0.0:4123 (Press CTRL+C to quit)
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:52628 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:40926 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:35834 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:40854 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:42862 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:53826 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:43900 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:41536 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:53804 - "GET /health HTTP/1.1" 200 OK
chatterbox-tts-api-gpu   | INFO:     127.0.0.1:34198 - "GET /health HTTP/1.1" 200 OK
Fetching 6 files: 100%|██████████| 6/6 [04:38<00:00, 46.42s/it]
chatterbox-tts-api-gpu   | ✗ Failed to initialize model: No CUDA GPUs are available
```

So as there is a blackwell build, it sadly results in the same error.

So I tried updating the Cuda runtime, set python3 instead of python3.11 as that was not available for ubuntu24.04.
Here is my DockerFile:

``` DockerFile
# Use NVIDIA CUDA runtime as base for better GPU support
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Make build ARG available in this build stage for RUN instructions
# Fuck uv, we use normal pip now
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Create and activate virtual environment
ENV VIRTUAL_ENV=/app/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN echo "Testing if virtual environment is activated!"
RUN which python3 && sleep 5

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --no-cache-dir --upgrade pip 

RUN python3 -m pip install --no-cache-dir setuptools wheel


# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify torch installation and CUDA availability
RUN echo "Testing torch installation and CUDA availability!"
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"

# Install base dependencies first
RUN pip3 install --no-cache-dir \
    fastapi \
    psutil \
    pydub \
    python-dotenv \
    python-multipart \
    requests \
    setuptools \
    sse-starlette



# The chatterbox source is *not* installed at build time anymore. Instead we
# install at container start so a host-mounted local copy (e.g. third_party/chatterbox)
# can be used directly without copying into the image. If no local copy is
# mounted the entrypoint will fall back to installing from the configured repo.
# ARG CHATTERBOX_REPO=https://github.com/resemble-ai/chatterbox.git

# Note: keep a final torch install line to ensure wheels are present in venv
# Note: This line can be removed if the initial install is sufficient,
# but can be useful for ensuring the correct versions are present.
# This line is redundant. The main torch installation is now handled above with the correct CUDA version.

# Copy application code
COPY app/ ./app/
COPY main.py ./

# Copy voice sample if it exists (optional, can be mounted)
COPY voice-sample.mp3 ./voice-sample.mp3

# Create directories for model cache and voice library (separate from source code)
RUN mkdir -p /cache /voices /data/long_text_jobs

# Set default environment variables (prefer CUDA)
ENV PORT=4123
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000
ENV DEVICE=cuda
ENV MODEL_CACHE_DIR=/cache
ENV VOICE_LIBRARY_DIR=/voices
ENV HOST=0.0.0.0

# Long text TTS settings
ENV LONG_TEXT_DATA_DIR=/data/long_text_jobs
ENV LONG_TEXT_MAX_LENGTH=100000
ENV LONG_TEXT_CHUNK_SIZE=2500
ENV LONG_TEXT_SILENCE_PADDING_MS=200
ENV LONG_TEXT_JOB_RETENTION_DAYS=7
ENV LONG_TEXT_MAX_CONCURRENT_JOBS=3

# NVIDIA/CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Copy entrypoint that will install chatterbox at container start (from a
# mounted local folder if available, otherwise from the repo URL)
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Use the entrypoint (it will exec the CMD)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command: run the app
CMD ["python", "main.py"]
```

And the changed blackwell.yml:

```yml
version: '3.8'

services:
  # Main API Service (always included)
  chatterbox-tts:
    build:
      context: ..
      dockerfile: docker/Dockerfile.blackwell
    container_name: chatterbox-tts-api-blackwell
    # Request GPUs from the Docker daemon so the container can see CUDA devices
    # Works with modern Docker Compose (docker compose) and the 'gpus' key.
    ports:
      - '${PORT:-4123}:${PORT:-4123}'
    environment:
      # API Configuration
      - PORT=${PORT:-4123}
      - HOST=${HOST:-0.0.0.0}

      # TTS Model Settings
      - EXAGGERATION=${EXAGGERATION:-0.5}
      - CFG_WEIGHT=${CFG_WEIGHT:-0.5}
      - TEMPERATURE=${TEMPERATURE:-0.8}

      # Text Processing
      - MAX_CHUNK_LENGTH=${MAX_CHUNK_LENGTH:-280}
      - MAX_TOTAL_LENGTH=${MAX_TOTAL_LENGTH:-3000}

      # Voice and Model Settings
      - VOICE_SAMPLE_PATH=/app/voice-sample.mp3
      - DEVICE=${DEVICE:-cuda}
      - MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/cache}
      - VOICE_LIBRARY_DIR=${VOICE_LIBRARY_DIR:-/voices}

      # NVIDIA/CUDA settings
    volumes:
      # Mount voice sample file (optional)
      - ${VOICE_SAMPLE_HOST_PATH:-../voice-sample.mp3}:/app/voice-sample.mp3:ro

      # Mount local chatterbox source for development
      - ../third_party/chatterbox:/app/third_party/chatterbox

      # Mount local models directory for persistence
      - ../models:${MODEL_CACHE_DIR:-/cache}

      # Mount voice library for persistence
      - chatterbox-voices:${VOICE_LIBRARY_DIR:-/voices}

      # Optional: Mount custom voice samples directory (legacy)
      - ${VOICE_SAMPLES_DIR:-../voice-samples}:/app/voice-samples:ro

    # GPU support (enabled by default for this compose file)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
# do not restart on failure to avoid rapid crash loops
    restart: "no"
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:${PORT:-4123}/health']
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s

  # Frontend Service with integrated proxy (optional - requires 'frontend' profile)
  frontend:
    profiles: ['frontend', 'ui', 'fullstack']
    build:
      context: ../frontend
      dockerfile: Dockerfile
    container_name: chatterbox-tts-frontend
    ports:
      - '${FRONTEND_PORT:-4321}:80' # Frontend serves on port 80 internally
    depends_on:
      - chatterbox-tts
    restart: unless-stopped

volumes:
  chatterbox-models:
    driver: local
  chatterbox-voices:
    driver: local
  chatterbox-longtext-data:
    driver: local

```



Still nothing, I even tried not using uv as you can see.

Next up would be to use the new Torch nightly, but for that I have to first compile it, will continue later.



