from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.schemas import PredictionResponse, SceneObject, SceneResponse


def build_scene_response(
    prediction: PredictionResponse,
    *,
    frame_id: int,
    camera_id: str,
    inference_ms: float | None = None,
    loop_fps: float | None = None,
) -> SceneResponse:
    return SceneResponse(
        frame_id=frame_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        camera_id=camera_id,
        model=prediction.model,
        image_size=prediction.image_size,
        inference_ms=None if inference_ms is None else round(inference_ms, 2),
        loop_fps=None if loop_fps is None else round(loop_fps, 2),
        objects=[
            SceneObject(
                id=detected.id,
                track_id=detected.track_id,
                label=detected.label,
                confidence=detected.confidence,
                bbox_xyxy=detected.bbox_xyxy,
                center_pixel=detected.center_pixel,
                area_pixels=detected.area_pixels,
                status=detected.status,
            )
            for detected in prediction.objects
        ],
    )


def write_scene_json(scene: SceneResponse, output_path: Path, *, retries: int = 3) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.{uuid4().hex}.tmp")
    temp_path.write_text(scene.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for attempt in range(retries + 1):
        try:
            temp_path.replace(output_path)
            return True
        except PermissionError:
            if attempt >= retries:
                temp_path.unlink(missing_ok=True)
                return False
            time.sleep(0.01 * (attempt + 1))

    temp_path.unlink(missing_ok=True)
    return False


def append_scene_jsonl(scene: SceneResponse, log_path: Path) -> bool:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(scene.model_dump_json() + "\n")
        return True
    except OSError:
        return False
