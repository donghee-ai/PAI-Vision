from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

from app.schemas import DetectedObject, PredictionResponse


def _polygon_centroid(points: list[list[int]], bbox_xyxy: list[float]) -> list[int]:
    if len(points) < 3:
        return [
            int(round((bbox_xyxy[0] + bbox_xyxy[2]) / 2)),
            int(round((bbox_xyxy[1] + bbox_xyxy[3]) / 2)),
        ]

    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, current in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        cross = current[0] * nxt[1] - nxt[0] * current[1]
        twice_area += cross
        cx += (current[0] + nxt[0]) * cross
        cy += (current[1] + nxt[1]) * cross

    if abs(twice_area) < 1e-6:
        return [
            int(round((bbox_xyxy[0] + bbox_xyxy[2]) / 2)),
            int(round((bbox_xyxy[1] + bbox_xyxy[3]) / 2)),
        ]

    return [int(round(cx / (3 * twice_area))), int(round(cy / (3 * twice_area)))]


@dataclass(frozen=True)
class YoloSegmentationService:
    model_path: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_model", YOLO(self.model_path))

    def predict(
        self,
        image: Image.Image,
        *,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> PredictionResponse:
        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        objects: list[DetectedObject] = []
        if boxes is None:
            return PredictionResponse(
                model=self.model_path,
                image_size=[image.width, image.height],
                objects=objects,
            )

        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            confidence = round(float(boxes.conf[idx].item()), 4)
            bbox_xyxy = [round(float(value), 2) for value in boxes.xyxy[idx].tolist()]

            polygon: list[list[int]] = []
            area_pixels: int | None = None
            if masks is not None and len(masks.xy) > idx:
                polygon = [
                    [int(round(point[0])), int(round(point[1]))]
                    for point in masks.xy[idx].tolist()
                ]
                if masks.data is not None and len(masks.data) > idx:
                    area_pixels = int(masks.data[idx].sum().item())

            objects.append(
                DetectedObject(
                    id=f"obj_{idx + 1:02d}",
                    label=result.names.get(cls_id, str(cls_id)),
                    confidence=confidence,
                    bbox_xyxy=bbox_xyxy,
                    mask_polygon=polygon,
                    center_pixel=_polygon_centroid(polygon, bbox_xyxy),
                    area_pixels=area_pixels,
                )
            )

        return PredictionResponse(
            model=self.model_path,
            image_size=[image.width, image.height],
            objects=objects,
        )
