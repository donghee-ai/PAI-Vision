from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    id: str
    track_id: str | None = None
    label: str
    confidence: float
    bbox_xyxy: list[float] = Field(description="[x1, y1, x2, y2]")
    mask_polygon: list[list[int]]
    center_pixel: list[int]
    area_pixels: int | None = None
    status: str = "detected"


class PredictionResponse(BaseModel):
    model: str
    image_size: list[int] = Field(description="[width, height]")
    objects: list[DetectedObject]


class SceneObject(BaseModel):
    id: str
    track_id: str | None = None
    label: str
    confidence: float
    bbox_xyxy: list[float] = Field(description="[x1, y1, x2, y2]")
    center_pixel: list[int]
    area_pixels: int | None = None
    status: str = "detected"


class SceneResponse(BaseModel):
    frame_id: int
    timestamp: str
    camera_id: str
    model: str
    image_size: list[int] = Field(description="[width, height]")
    inference_ms: float | None = None
    loop_fps: float | None = None
    objects: list[SceneObject]
