from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yolo_model: str = Field(default="yolo11s-seg.pt", alias="YOLO_MODEL")
    yolo_device: str = Field(default="cpu", alias="YOLO_DEVICE")
    yolo_imgsz: int = Field(default=640, alias="YOLO_IMGSZ")
    yolo_conf: float = Field(default=0.25, alias="YOLO_CONF")
    yolo_iou: float = Field(default=0.7, alias="YOLO_IOU")
    camera_id: str = Field(default="front_rgb", alias="CAMERA_ID")
    camera_index: int = Field(default=0, alias="CAMERA_INDEX")
    camera_target_fps: float = Field(default=10.0, alias="CAMERA_TARGET_FPS")
    camera_width: int = Field(default=1280, alias="CAMERA_WIDTH")
    camera_height: int = Field(default=720, alias="CAMERA_HEIGHT")
    scene_json_path: str = Field(default="runtime/latest_scene.json", alias="SCENE_JSON_PATH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
