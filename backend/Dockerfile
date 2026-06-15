FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install ultralytics first (pulls opencv-python), then swap to headless variant
RUN pip install --no-cache-dir ultralytics && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless && \
    pip install --no-cache-dir fastapi==0.111.0 "uvicorn[standard]==0.29.0" python-multipart==0.0.9
# Model downloads on first request (avoids build-time weight compat issues)

COPY main.py .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
