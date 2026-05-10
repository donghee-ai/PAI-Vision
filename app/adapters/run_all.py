from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import threading
import time

import uvicorn

from app.config import get_settings
from app.perception.live_camera import LiveCameraConfig, run_live_camera


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run live camera inference with the local API/WebSocket development adapter."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for local API adapter")
    parser.add_argument("--port", type=int, default=8000, help="Port for local API adapter")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload (dev only)")
    parser.add_argument(
        "--server-wait-seconds",
        "--adapter-wait-seconds",
        dest="adapter_wait_seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after starting local API adapter before camera loop",
    )

    parser.add_argument("--camera", type=int, default=settings.camera_index)
    parser.add_argument("--camera-id", default=settings.camera_id)
    parser.add_argument("--width", type=int, default=settings.camera_width)
    parser.add_argument("--height", type=int, default=settings.camera_height)
    parser.add_argument("--target-fps", type=float, default=settings.camera_target_fps)
    parser.add_argument("--model", default=settings.yolo_model)
    parser.add_argument("--device", default=settings.yolo_device)
    parser.add_argument("--imgsz", type=int, default=settings.yolo_imgsz)
    parser.add_argument("--conf", type=float, default=settings.yolo_conf)
    parser.add_argument("--iou", type=float, default=settings.yolo_iou)
    parser.add_argument("--scene-json", type=Path, default=Path(settings.scene_json_path))
    parser.add_argument("--scene-log-dir", type=Path, default=Path(settings.scene_log_dir))
    parser.add_argument("--no-scene-json", action="store_true")
    parser.add_argument("--no-session-log", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-name", default="PAI-Vision Live")
    return parser.parse_args()



def _run_api_adapter(host: str, port: int, reload: bool) -> None:
    config = uvicorn.Config("app.adapters.local_api:app", host=host, port=port, reload=reload, log_level="info")
    adapter = uvicorn.Server(config)
    asyncio.run(adapter.serve())



def _build_live_camera_config(args: argparse.Namespace) -> LiveCameraConfig:
    return LiveCameraConfig(
        camera=args.camera,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        scene_json=Path(args.scene_json),
        no_scene_json=args.no_scene_json,
        scene_log_dir=Path(args.scene_log_dir),
        no_session_log=args.no_session_log,
        no_display=args.no_display,
        max_frames=args.max_frames,
        window_name=args.window_name,
    )



def main() -> None:
    args = parse_args()

    print(f"Starting local API/WebSocket adapter on http://{args.host}:{args.port}")
    adapter_thread = threading.Thread(
        target=_run_api_adapter,
        args=(args.host, args.port, args.reload),
        daemon=True,
    )
    adapter_thread.start()

    if args.adapter_wait_seconds > 0:
        time.sleep(args.adapter_wait_seconds)

    print("Starting live camera loop with shared scene JSON output")
    run_live_camera(_build_live_camera_config(args))


if __name__ == "__main__":
    main()
