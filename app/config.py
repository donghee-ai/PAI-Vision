from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yolo_model: str = Field(default="yoloe-11s-seg.pt", alias="YOLO_MODEL")
    yolo_device: str = Field(default="auto", alias="YOLO_DEVICE")
    yolo_imgsz: int = Field(default=640, alias="YOLO_IMGSZ")
    yolo_conf: float = Field(default=0.25, alias="YOLO_CONF")
    yolo_iou: float = Field(default=0.7, alias="YOLO_IOU")
    # Open-vocabulary prompts. Comma-separated. Only used by open-vocab models
    # (YOLOE / YOLO-World); ignored by closed-vocab models (e.g. yolo11s-seg).
    # YOLOE-seg detects NOTHING until these are set, so a usable default is given.
    yolo_classes: str | None = Field(
        default="person,orange,plate,cup,bottle,bowl,banana,apple,box,scissors",
        alias="YOLO_CLASSES",
    )
    camera_id: str = Field(default="front_rgb", alias="CAMERA_ID")
    camera_index: int = Field(default=0, alias="CAMERA_INDEX")
    cameras: str | None = Field(default=None, alias="CAMERAS")
    camera_target_fps: float = Field(default=10.0, alias="CAMERA_TARGET_FPS")
    camera_capture_fps: float = Field(default=30.0, alias="CAMERA_CAPTURE_FPS")
    camera_width: int = Field(default=1280, alias="CAMERA_WIDTH")
    camera_height: int = Field(default=720, alias="CAMERA_HEIGHT")
    scene_json_path: str = Field(default="runtime/latest_scene.json", alias="SCENE_JSON_PATH")
    scene_log_dir: str = Field(default="runtime/logs", alias="SCENE_LOG_DIR")
    zmq_publish_enabled: bool = Field(default=False, alias="ZMQ_PUBLISH_ENABLED")
    zmq_publish_bind: str = Field(default="tcp://*:5555", alias="ZMQ_PUBLISH_BIND")
    zmq_publish_fps: float = Field(default=30.0, alias="ZMQ_PUBLISH_FPS")
    zmq_publish_jpeg_quality: int = Field(default=90, alias="ZMQ_PUBLISH_JPEG_QUALITY")

    # ─── Foreground cutout stream (background removed, objects only) ──────────
    # Builds a per-frame RGBA cutout from the detected-object masks (background
    # made transparent) and streams it over WebSocket (/ws/cutouts) and,
    # optionally, a dedicated ZMQ PUB socket. Encoding runs on its own thread so
    # it never charges the YOLO loop budget.
    cutout_enabled: bool = Field(default=True, alias="CUTOUT_ENABLED")
    cutout_fps: float = Field(default=15.0, alias="CUTOUT_FPS")
    cutout_format: str = Field(default="webp", alias="CUTOUT_FORMAT")  # webp | png
    cutout_quality: int = Field(default=80, alias="CUTOUT_QUALITY")  # webp lossy quality
    cutout_zmq_enabled: bool = Field(default=False, alias="CUTOUT_ZMQ_ENABLED")
    cutout_zmq_bind: str = Field(default="tcp://*:5556", alias="CUTOUT_ZMQ_BIND")


@lru_cache
def get_settings() -> Settings:
    return Settings()
