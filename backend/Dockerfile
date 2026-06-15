FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default human pose model so first request isn't slow.
# When dog weights are available, mount them via volume and set MODEL_PATH env var.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY main.py .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
