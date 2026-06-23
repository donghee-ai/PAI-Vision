from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from app.perception.device import resolve_yolo_device
from app.perception.schemas import DetectedObject, PredictionResponse

logger = logging.getLogger(__name__)


def _parse_classes(classes: str | None) -> list[str]:
    if not classes:
        return []
    return [name.strip() for name in classes.split(",") if name.strip()]


PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 184, 148),
    (9, 132, 227),
    (253, 203, 110),
    (214, 48, 49),
    (108, 92, 231),
    (232, 67, 147),
)


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


def render_prediction_overlay(
    image: Image.Image,
    prediction: PredictionResponse,
    *,
    mask_alpha: int = 80,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(3, round(max(base.size) / 500))

    for idx, detected in enumerate(prediction.objects):
        color = PALETTE[idx % len(PALETTE)]
        rgba = (*color, mask_alpha)

        if len(detected.mask_polygon) >= 3:
            polygon = [tuple(point) for point in detected.mask_polygon]
            overlay_draw.polygon(polygon, fill=rgba)

    rendered = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(rendered)

    for idx, detected in enumerate(prediction.objects):
        color = PALETTE[idx % len(PALETTE)]

        if len(detected.mask_polygon) >= 3:
            polygon = [tuple(point) for point in detected.mask_polygon]
            draw.line(polygon + [polygon[0]], fill=(*color, 255), width=line_width)

        x1, y1, x2, y2 = [int(round(value)) for value in detected.bbox_xyxy]
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=line_width)

        center_x, center_y = detected.center_pixel
        marker = max(6, line_width * 3)
        draw.line(
            (center_x - marker, center_y, center_x + marker, center_y),
            fill=(*color, 255),
            width=line_width,
        )
        draw.line(
            (center_x, center_y - marker, center_x, center_y + marker),
            fill=(*color, 255),
            width=line_width,
        )

        object_id = detected.track_id or detected.id
        label = f"{object_id} {detected.label} {detected.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 8)
        draw.rectangle(
            (x1, label_y, x1 + text_w + 8, label_y + text_h + 6),
            fill=(*color, 230),
        )
        draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255, 255), font=font)

    return rendered.convert("RGB")


@dataclass(frozen=True)
class YoloSegmentationService:
    model_path: str
    device: str
    classes: str | None = None

    def __post_init__(self) -> None:
        model = YOLO(self.model_path)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_resolved_device", resolve_yolo_device(self.device))
        self._apply_open_vocabulary(model)

    def _apply_open_vocabulary(self, model: YOLO) -> None:
        """Set open-vocabulary prompts on YOLOE / YOLO-World models.

        YOLOE exposes ``get_text_pe`` and needs the precomputed text embeddings;
        YOLO-World embeds the names internally via ``set_classes`` alone. Closed-
        vocabulary models (e.g. yolo11s-seg) have neither and keep their trained
        labels — a configured class list is then a no-op (with a warning).
        """
        names = _parse_classes(self.classes)
        if not names:
            return

        if hasattr(model, "get_text_pe"):  # YOLOE
            model.set_classes(names, model.get_text_pe(names))
            logger.info("YOLOE open vocabulary set to %d classes: %s", len(names), names)
        elif hasattr(model, "set_classes"):  # YOLO-World
            model.set_classes(names)
            logger.info("YOLO-World open vocabulary set to %d classes: %s", len(names), names)
        else:
            logger.warning(
                "Model %s is closed-vocabulary; ignoring YOLO_CLASSES=%s",
                self.model_path,
                self.classes,
            )

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

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
            device=self._resolved_device,
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
