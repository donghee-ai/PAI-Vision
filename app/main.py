from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import PredictionResponse
from app.vision import YoloSegmentationService

app = FastAPI(title="PAI-Vision YOLO Segmentation Server", version="0.1.0")


@lru_cache
def get_service() -> YoloSegmentationService:
    settings = get_settings()
    return YoloSegmentationService(
        model_path=settings.yolo_model,
        device=settings.yolo_device,
    )


@app.get("/health")
def health() -> dict[str, str | int | float]:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.yolo_model,
        "device": settings.yolo_device,
        "imgsz": settings.yolo_imgsz,
        "conf": settings.yolo_conf,
        "iou": settings.yolo_iou,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> PredictionResponse:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )
