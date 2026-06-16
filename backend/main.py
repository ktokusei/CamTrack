from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
import torch
import pathlib

# Locate index.html: Docker puts it alongside main.py (/app/); on Mac it's one level up
_here = pathlib.Path(__file__).resolve().parent
_candidates = [_here / "index.html", _here.parent / "index.html"]
INDEX_HTML = next((str(p) for p in _candidates if p.exists()), "/app/index.html")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MODEL_PATH can be overridden via env var to point at dog-pose weights
# once trained. Falls back to human pose model so the API is always usable.
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n-pose.pt")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "human")  # "human" or "dog"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Loading model: {MODEL_PATH} (type={MODEL_TYPE}, device={DEVICE})")
model = YOLO(MODEL_PATH)
print("Model ready.")

# Keypoint name lists per model type.
# Human: COCO 17-keypoint standard
HUMAN_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Dog: Ultralytics Dog-Pose 24-keypoint schema
DOG_KP_NAMES = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end",
    "left_ear_base", "right_ear_base",
    "nose", "chin",
    "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye",
    "withers", "throat",
]

KP_NAMES = DOG_KP_NAMES if MODEL_TYPE == "dog" else HUMAN_KP_NAMES


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "type": MODEL_TYPE}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    results = model(img, verbose=False, device=DEVICE)
    keypoints = []

    if results and results[0].keypoints is not None:
        kps = results[0].keypoints
        if kps.xy is not None and len(kps.xy) > 0:
            h, w = img.shape[:2]
            xy   = kps.xy[0].cpu().numpy()    # (N, 2)  pixel coords
            conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

            for i, (x, y) in enumerate(xy):
                name = KP_NAMES[i] if i < len(KP_NAMES) else f"kp_{i}"
                keypoints.append({
                    "index": i,
                    "name": name,
                    "x": float(x) / w,   # normalised 0-1
                    "y": float(y) / h,
                    "confidence": float(conf[i]),
                })

    return {"keypoints": keypoints, "model_type": MODEL_TYPE}
