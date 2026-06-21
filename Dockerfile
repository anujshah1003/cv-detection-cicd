# Slim Python base keeps the image small. We don't need a CUDA image
# since yolov8n is light enough to run acceptably on CPU for a demo.
FROM python:3.11-slim

# System deps required by opencv-python (a transitive dependency of
# ultralytics) to handle image I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements first and install separately from app code.
# Docker caches layers -- if requirements.txt hasn't changed, this layer
# is reused on rebuild instead of reinstalling everything, which makes
# CI/CD rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time, not at container startup.
# This means the container starts instantly in production instead of
# pausing to download weights on first request.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY app/ ./app/

# Hugging Face Spaces expects the app to listen on port 7860.
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
