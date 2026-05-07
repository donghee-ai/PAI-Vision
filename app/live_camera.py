from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.scene import append_scene_jsonl, build_scene_response, write_scene_json
from app.tracking import CentroidTracker
from app.vision import YoloSegmentationService, render_prediction_overlay


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run realtime YOLO segmentation from a camera.")
    parser.add_argument("--camera", type=int, default=settings.camera_index, help="OpenCV camera index")
    parser.add_argument("--camera-id", default=settings.camera_id, help="Camera id for scene JSON")
    parser.add_argument("--width", type=int, default=settings.camera_width, help="Requested capture width")
    parser.add_argument("--height", type=int, default=settings.camera_height, help="Requested capture height")
    parser.add_argument("--target-fps", type=float, default=settings.camera_target_fps, help="Target processing FPS")
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


def main() -> None:
    args = parse_args()
    target_interval = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    service = YoloSegmentationService(model_path=args.model, device=args.device)
    tracker = CentroidTracker()
    capture = open_camera(args.camera, args.width, args.height)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_log_path = args.scene_log_dir / f"live_camera_{session_id}.jsonl"

    frame_id = 0
    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    scene_write_ok = True
    log_write_ok = True

    print(
        f"Starting live camera: camera={args.camera}, device={args.device} -> {service.resolved_device}, "
        f"model={args.model}, target_fps={args.target_fps}"
    )
    print(f"Session id: {session_id}")
    if not args.no_session_log:
        print(f"Session log: {scene_log_path}")
    print("Press q or ESC in the display window to quit.")

    try:
        while True:
            loop_started = time.perf_counter()
            ok, frame_bgr = capture.read()
            if not ok:
                print("Camera frame read failed; retrying...")
                time.sleep(0.1)
                continue

            frame_id += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            inference_started = time.perf_counter()
            prediction = service.predict(
                image,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
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
                camera_id=args.camera_id,
                inference_ms=inference_ms,
                loop_fps=smoothed_fps,
            )
            if not args.no_scene_json:
                scene_write_ok = write_scene_json(scene, args.scene_json)
            if not args.no_session_log:
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
            cv2.putText(
                rendered_bgr,
                status,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered_bgr,
                status,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not args.no_display:
                cv2.imshow(args.window_name, rendered_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames is not None and frame_id >= args.max_frames:
                break

            elapsed = time.perf_counter() - loop_started
            if target_interval > elapsed:
                time.sleep(target_interval - elapsed)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
