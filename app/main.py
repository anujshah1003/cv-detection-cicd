"""
FastAPI service exposing object detection over HTTP.

POST /detect with an image file returns detected objects as JSON.
This is the service that gets containerized and deployed.
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.detector import ObjectDetector

app = FastAPI(
    title="CV Object Detection API",
    description="Small YOLOv8-based object detection demo with a full CI/CD pipeline.",
    version="1.0.0",
)

# Loaded once at startup, reused across requests (loading a model per
# request would be extremely slow).
detector = ObjectDetector()


@app.get("/")
def root():
    return {"status": "ok", "message": "Object detection API is running. POST an image to /detect"}


@app.get("/health")
def health():
    """Used by deployment platforms to check the service is alive."""
    return {"status": "healthy"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detections = detector.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse(
        {
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "box": [round(c, 2) for c in d.box],
                }
                for d in detections
            ],
        }
    )
