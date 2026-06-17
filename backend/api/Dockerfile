FROM python:3.11-slim

WORKDIR /app

# System libs needed by mediapipe and opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch (separate layer so it's cached between rebuilds) ──────────
# Saves ~1.5 GB vs the default CUDA wheels
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining backend dependencies ────────────────────────────────────────────
COPY requirements-docker.txt requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── Project source ─────────────────────────────────────────────────────────────
# datasets/ is baked in (label files). data/ and models/ are mounted at runtime.
COPY api/        api/
COPY hand_model/ hand_model/
COPY scripts/    scripts/
COPY datasets/   datasets/
COPY pyproject.toml .

# Runtime directories (overridden by bind-mounts in production)
RUN mkdir -p data/landmarks data/sequences \
             models/classifier models/sequence models/exports

# Install the api + hand_model packages so imports resolve inside the container
RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
