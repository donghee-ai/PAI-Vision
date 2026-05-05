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


@lru_cache
def get_settings() -> Settings:
    return Settings()
