from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import uvicorn

from app.adapters.cutout_stream import CutoutStreamer, cutout_bus
from app.adapters.scene_bus import scene_bus
from app.adapters.zmq_frame_publisher import ZmqFramePublisher
from app.config import get_settings
from app.perception.live_camera import LiveCameraConfig, run_live_camera


@dataclass(frozen=True)
class CameraRegistration:
    camera_id: str
    camera_index: int


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
    parser.add_argument(
        "--cameras",
        default=settings.cameras,
        help=(
            "Comma-separated camera registrations as camera_id:index. "
            "Example: front_rgb:0,side_rgb:1. Defaults to --camera-id/--camera."
        ),
    )
    parser.add_argument("--width", type=int, default=settings.camera_width)
    parser.add_argument("--height", type=int, default=settings.camera_height)
    parser.add_argument("--target-fps", type=float, default=settings.camera_target_fps, help="YOLO/perception FPS cap")
    parser.add_argument("--capture-fps", type=float, default=settings.camera_capture_fps, help="Capture thread FPS (raw + ZMQ)")
    parser.add_argument("--model", default=settings.yolo_model)
    parser.add_argument("--device", default=settings.yolo_device)
    parser.add_argument("--classes", default=settings.yolo_classes, help="Comma-separated open-vocab prompts (YOLOE/YOLO-World)")
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    scene_bus.attach_loop(loop)
    cutout_bus.attach_loop(loop)

    config = uvicorn.Config("app.adapters.local_api:app", host=host, port=port, reload=reload, log_level="info")
    adapter = uvicorn.Server(config)
    loop.run_until_complete(adapter.serve())



def _parse_camera_registrations(args: argparse.Namespace) -> list[CameraRegistration]:
    if not args.cameras:
        return [CameraRegistration(camera_id=args.camera_id, camera_index=args.camera)]

    registrations: list[CameraRegistration] = []
    for raw_spec in args.cameras.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue

        if ":" not in spec:
            raise ValueError(f"Invalid camera spec {spec!r}; expected camera_id:index")

        camera_id, index_text = spec.split(":", 1)
        camera_id = camera_id.strip()
        index_text = index_text.strip()
        if not camera_id:
            raise ValueError(f"Invalid camera spec {spec!r}; camera_id is empty")
        try:
            camera_index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid camera spec {spec!r}; index must be an integer") from exc

        registrations.append(CameraRegistration(camera_id=camera_id, camera_index=camera_index))

    if not registrations:
        raise ValueError("--cameras did not contain any camera registrations")

    camera_ids = [registration.camera_id for registration in registrations]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"Camera ids must be unique: {', '.join(camera_ids)}")
    return registrations


def _safe_camera_id(camera_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in camera_id)


def _scene_json_path(base_path: Path, camera_id: str, camera_count: int) -> Path:
    path_text = str(base_path)
    if "{camera_id}" in path_text:
        return Path(path_text.format(camera_id=_safe_camera_id(camera_id)))
    if camera_count == 1:
        return base_path
    suffix = base_path.suffix or ".json"
    return base_path.with_name(f"{base_path.stem}_{_safe_camera_id(camera_id)}{suffix}")


def _build_live_camera_config(
    args: argparse.Namespace,
    registration: CameraRegistration,
    camera_count: int,
) -> LiveCameraConfig:
    return LiveCameraConfig(
        camera=registration.camera_index,
        camera_id=registration.camera_id,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        capture_fps=args.capture_fps,
        model=args.model,
        device=args.device,
        classes=args.classes,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        scene_json=_scene_json_path(Path(args.scene_json), registration.camera_id, camera_count),
        no_scene_json=args.no_scene_json,
        scene_log_dir=Path(args.scene_log_dir),
        no_session_log=args.no_session_log,
        no_display=args.no_display,
        max_frames=args.max_frames,
        window_name=args.window_name if camera_count == 1 else f"{args.window_name} [{registration.camera_id}]",
    )


def _run_camera_worker(
    config: LiveCameraConfig,
    frame_publisher: ZmqFramePublisher | None = None,
    cutout_streamer: CutoutStreamer | None = None,
) -> None:
    print(f"Starting camera worker {config.camera_id} on OpenCV index {config.camera}")
    on_frame = frame_publisher.publish if frame_publisher is not None and frame_publisher.enabled else None
    on_cutout = cutout_streamer.publish if cutout_streamer is not None and cutout_streamer.enabled else None
    run_live_camera(
        config,
        on_scene=scene_bus.publish_nowait,
        on_frame=on_frame,
        on_cutout=on_cutout,
    )



def main() -> None:
    args = parse_args()
    settings = get_settings()
    camera_registrations = _parse_camera_registrations(args)

    print(f"Starting local API/WebSocket adapter on http://{args.host}:{args.port}")
    adapter_thread = threading.Thread(
        target=_run_api_adapter,
        args=(args.host, args.port, args.reload),
        daemon=True,
    )
    adapter_thread.start()

    if args.adapter_wait_seconds > 0:
        time.sleep(args.adapter_wait_seconds)

    frame_publisher: ZmqFramePublisher | None = None
    if settings.zmq_publish_enabled:
        frame_publisher = ZmqFramePublisher(
            bind_address=settings.zmq_publish_bind,
            fps=settings.zmq_publish_fps,
            jpeg_quality=settings.zmq_publish_jpeg_quality,
        )
        frame_publisher.start()
        if frame_publisher.enabled:
            print(
                f"ZMQ raw-frame publisher active at {settings.zmq_publish_bind} "
                f"(fps={settings.zmq_publish_fps}, jpeg_q={settings.zmq_publish_jpeg_quality})"
            )

    cutout_streamer: CutoutStreamer | None = None
    if settings.cutout_enabled:
        cutout_streamer = CutoutStreamer(
            fps=settings.cutout_fps,
            image_format=settings.cutout_format,
            quality=settings.cutout_quality,
            zmq_enabled=settings.cutout_zmq_enabled,
            zmq_bind=settings.cutout_zmq_bind,
        )
        cutout_streamer.start()
        if cutout_streamer.enabled:
            zmq_note = f", ZMQ {settings.cutout_zmq_bind}" if settings.cutout_zmq_enabled else ""
            print(
                f"Cutout streamer active -> WebSocket /ws/cutouts "
                f"(fps={settings.cutout_fps}, format={settings.cutout_format}{zmq_note})"
            )

    camera_configs = [
        _build_live_camera_config(args, registration, len(camera_registrations))
        for registration in camera_registrations
    ]
    print(f"Starting {len(camera_configs)} live camera worker(s) with direct scene push to local adapter")
    camera_threads = [
        threading.Thread(
            target=_run_camera_worker,
            args=(config, frame_publisher, cutout_streamer),
            name=f"camera-{config.camera_id}",
            daemon=True,
        )
        for config in camera_configs
    ]

    for camera_thread in camera_threads:
        camera_thread.start()

    try:
        for camera_thread in camera_threads:
            camera_thread.join()
    except KeyboardInterrupt:
        print("Stopping after keyboard interrupt; camera workers will exit with the process.")
    finally:
        if frame_publisher is not None:
            frame_publisher.stop()
        if cutout_streamer is not None:
            cutout_streamer.stop()


if __name__ == "__main__":
    main()
