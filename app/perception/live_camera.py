from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.perception.frame_buffer import FrameBuffer
from app.perception.scene import append_scene_jsonl, build_scene_response, write_scene_json
from app.perception.tracking import CentroidTracker
from app.perception.vision import YoloSegmentationService, render_prediction_overlay


@dataclass
class LiveCameraConfig:
    camera: int
    camera_id: str
    width: int
    height: int
    target_fps: float
    capture_fps: float
    model: str
    device: str
    imgsz: int
    conf: float
    iou: float
    scene_json: Path
    no_scene_json: bool
    scene_log_dir: Path
    no_session_log: bool
    no_display: bool
    max_frames: int | None
    window_name: str



def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run realtime YOLO segmentation from a camera.")
    parser.add_argument("--camera", type=int, default=settings.camera_index, help="OpenCV camera index")
    parser.add_argument("--camera-id", default=settings.camera_id, help="Camera id for scene JSON")
    parser.add_argument("--width", type=int, default=settings.camera_width, help="Requested capture width")
    parser.add_argument("--height", type=int, default=settings.camera_height, help="Requested capture height")
    parser.add_argument("--target-fps", type=float, default=settings.camera_target_fps, help="Target perception (YOLO) FPS")
    parser.add_argument("--capture-fps", type=float, default=settings.camera_capture_fps, help="Capture thread FPS (raw frames + ZMQ)")
    parser.add_argument("--model", default=settings.yolo_model, help="Model path or Ultralytics model name")
    parser.add_argument("--device", default=settings.yolo_device, help="Device, e.g. auto, cpu, mps, cuda:0")
    parser.add_argument("--imgsz", type=int, default=settings.yolo_imgsz, help="Inference image size")
    parser.add_argument("--conf", type=float, default=settings.yolo_conf, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=settings.yolo_iou, help="IoU threshold")
    parser.add_argument("--scene-json", type=Path, default=Path(settings.scene_json_path), help="Latest scene JSON path")
    parser.add_argument("--no-scene-json", action="store_true", help="Do not write latest scene JSON")
    parser.add_argument("--scene-log-dir", type=Path, default=Path(settings.scene_log_dir), help="Directory for session JSONL logs")
    parser.add_argument("--no-session-log", action="store_true", help="Do not write per-frame session JSONL log")
    parser.add_argument("--no-display", action="store_true", help="Run camera loop without opening a display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames")
    parser.add_argument("--window-name", default="PAI-Vision Live", help="Display window title")
    return parser.parse_args()



def config_from_args(args: argparse.Namespace) -> LiveCameraConfig:
    return LiveCameraConfig(
        camera=args.camera,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        capture_fps=args.capture_fps,
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



def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture



def run_live_camera(
    config: LiveCameraConfig,
    *,
    on_scene: Callable[[dict[str, Any]], None] | None = None,
    on_frame: Callable[[str, np.ndarray], None] | None = None,
) -> None:
    """Run live camera with capture and perception on independent threads.

    Capture thread (config.capture_fps) reads raw frames at the camera's native
    rate, hands them to `on_frame` (ZMQ publisher) immediately, and pushes the
    latest frame into a single-slot FrameBuffer. Perception thread (this thread,
    capped at config.target_fps) consumes from the buffer at its own pace —
    YOLO slowdowns therefore never throttle raw-frame streaming.
    """
    service = YoloSegmentationService(model_path=config.model, device=config.device)
    tracker = CentroidTracker()
    capture = open_camera(config.camera, config.width, config.height)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_log_path = config.scene_log_dir / f"live_camera_{config.camera_id}_{session_id}.jsonl"

    print(
        f"Starting live camera: camera={config.camera}, device={config.device} -> {service.resolved_device}, "
        f"camera_id={config.camera_id}, model={config.model}, "
        f"capture_fps={config.capture_fps}, target_fps={config.target_fps}"
    )
    print(f"Session id: {session_id}")
    if not config.no_session_log:
        print(f"Session log: {scene_log_path}")
    print("Press q or ESC in the display window to quit.")

    frame_buffer = FrameBuffer()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(capture, config, frame_buffer, stop_event, on_frame),
        name=f"capture-{config.camera_id}",
        daemon=True,
    )
    capture_thread.start()

    try:
        _perception_loop(
            config=config,
            service=service,
            tracker=tracker,
            frame_buffer=frame_buffer,
            stop_event=stop_event,
            scene_log_path=scene_log_path,
            on_scene=on_scene,
        )
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        capture.release()
        cv2.destroyAllWindows()


def _capture_loop(
    capture: cv2.VideoCapture,
    config: LiveCameraConfig,
    frame_buffer: FrameBuffer,
    stop_event: threading.Event,
    on_frame: Callable[[str, np.ndarray], None] | None,
) -> None:
    """Pull frames at the camera's native rate and fan out: ZMQ + perception buffer."""
    capture_interval = 1.0 / config.capture_fps if config.capture_fps > 0 else 0.0
    consecutive_failures = 0
    while not stop_event.is_set():
        loop_started = time.perf_counter()
        ok, frame_bgr = capture.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures <= 3 or consecutive_failures % 30 == 0:
                print(f"Camera {config.camera_id} frame read failed ({consecutive_failures})")
            if stop_event.wait(0.05):
                break
            continue
        consecutive_failures = 0

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if on_frame is not None:
            on_frame(config.camera_id, frame_rgb)
        frame_buffer.put(frame_rgb)

        elapsed = time.perf_counter() - loop_started
        remaining = capture_interval - elapsed
        if remaining > 0 and stop_event.wait(remaining):
            break


def _perception_loop(
    *,
    config: LiveCameraConfig,
    service: YoloSegmentationService,
    tracker: CentroidTracker,
    frame_buffer: FrameBuffer,
    stop_event: threading.Event,
    scene_log_path: Path,
    on_scene: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Consume latest frame, run YOLO, publish scene, render display."""
    target_interval = 1.0 / config.target_fps if config.target_fps > 0 else 0.0
    frame_id = 0
    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    scene_write_ok = True
    log_write_ok = True

    while not stop_event.is_set():
        loop_started = time.perf_counter()

        frame_rgb, _ = frame_buffer.get(timeout=1.0)
        if frame_rgb is None:
            continue

        frame_id += 1
        image = Image.fromarray(frame_rgb)

        inference_started = time.perf_counter()
        prediction = service.predict(
            image,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
        )
        prediction = tracker.update(prediction)
        inference_ms = (time.perf_counter() - inference_started) * 1000

        now = time.perf_counter()
        instantaneous_fps = 1.0 / max(now - last_loop_time, 1e-6)
        smoothed_fps = instantaneous_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * instantaneous_fps)
        last_loop_time = now

        scene = build_scene_response(
            prediction,
            frame_id=frame_id,
            camera_id=config.camera_id,
            inference_ms=inference_ms,
            loop_fps=smoothed_fps,
        )
        if on_scene is not None:
            on_scene(scene.model_dump())
        if not config.no_scene_json:
            scene_write_ok = write_scene_json(scene, config.scene_json)
        if not config.no_session_log:
            log_write_ok = append_scene_jsonl(scene, scene_log_path)

        rendered = render_prediction_overlay(image, prediction)
        rendered_bgr = cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)

        status = (
            f"frame {frame_id} | objects {len(scene.objects)} | "
            f"infer {inference_ms:.1f} ms | loop {smoothed_fps:.1f} fps"
        )
        if not scene_write_ok:
            status += " | scene write skipped"
        if not log_write_ok:
            status += " | log write skipped"
        cv2.putText(rendered_bgr, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rendered_bgr, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if not config.no_display:
            cv2.imshow(config.window_name, rendered_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()
                break

        if config.max_frames is not None and frame_id >= config.max_frames:
            stop_event.set()
            break

        elapsed = time.perf_counter() - loop_started
        remaining = target_interval - elapsed
        if remaining > 0 and stop_event.wait(remaining):
            break



def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_live_camera(config)


if __name__ == "__main__":
    main()
