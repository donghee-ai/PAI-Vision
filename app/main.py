from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.device import resolve_yolo_device
from app.scene import build_scene_response
from app.schemas import PredictionResponse, SceneResponse
from app.vision import YoloSegmentationService, render_prediction_overlay

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
        "resolved_device": resolve_yolo_device(settings.yolo_device),
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


@app.post("/predict/scene", response_model=SceneResponse)
async def predict_scene(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> SceneResponse:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    started = perf_counter()
    prediction = get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )
    inference_ms = (perf_counter() - started) * 1000
    return build_scene_response(
        prediction,
        frame_id=0,
        camera_id=settings.camera_id,
        inference_ms=inference_ms,
    )


@app.post("/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    prediction = get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )
    rendered = render_prediction_overlay(image, prediction)
    output = BytesIO()
    rendered.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")


@app.get("/scene/latest")
def latest_scene() -> Response:
    settings = get_settings()
    scene_path = Path(settings.scene_json_path)
    if not scene_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene JSON not found: {scene_path}")
    return Response(content=scene_path.read_text(encoding="utf-8"), media_type="application/json")
