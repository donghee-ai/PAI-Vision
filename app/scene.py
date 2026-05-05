from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.schemas import PredictionResponse, SceneObject, SceneResponse


def build_scene_response(
    prediction: PredictionResponse,
    *,
    frame_id: int,
    camera_id: str,
    inference_ms: float | None = None,
) -> SceneResponse:
    return SceneResponse(
        frame_id=frame_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        camera_id=camera_id,
        model=prediction.model,
        image_size=prediction.image_size,
        inference_ms=None if inference_ms is None else round(inference_ms, 2),
        objects=[
            SceneObject(
                id=detected.id,
                label=detected.label,
                confidence=detected.confidence,
                bbox_xyxy=detected.bbox_xyxy,
                center_pixel=detected.center_pixel,
                area_pixels=detected.area_pixels,
                depth_m=detected.depth_m,
                camera_xyz=detected.camera_xyz,
                robot_xyz=detected.robot_xyz,
                status=detected.status,
            )
            for detected in prediction.objects
        ],
    )


def write_scene_json(scene: SceneResponse, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(scene.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
