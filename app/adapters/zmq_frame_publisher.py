"""ZMQ camera frame publisher.

Publishes RGB camera frames as base64-encoded JPEG over a ZMQ PUB socket using
the same wire format as LeRobot's `image_server.py`, so a LeRobot policy can
consume the stream via `lerobot.cameras.zmq.ZMQCamera` without modification:

    {
        "timestamps": {"<camera_id>": float, ...},
        "images":     {"<camera_id>": "<base64-jpeg>", ...}
    }

Design notes
------------
- `publish(camera_id, frame_rgb)` is non-blocking. The caller (camera loop)
  just hands the latest frame reference into a per-camera slot under a lock.
- A single background thread runs at `fps` Hz, snapshots the slots, encodes
  JPEG, and sends one JSON message per tick. JPEG encoding therefore does NOT
  charge the YOLO loop budget.
- If `pyzmq` is not installed the publisher becomes a no-op, so the rest of
  the perception pipeline keeps running.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dep elsewhere in PAI-Vision
    cv2 = None  # type: ignore[assignment]

try:
    import zmq
except ImportError:  # pyzmq is optional
    zmq = None  # type: ignore[assignment]


class ZmqFramePublisher:
    """Background ZMQ PUB publisher for camera frames.

    Thread-safe: `publish()` may be called concurrently from multiple camera
    worker threads (one per camera_id).
    """

    def __init__(
        self,
        *,
        bind_address: str = "tcp://*:5555",
        fps: float = 30.0,
        jpeg_quality: int = 90,
        sndhwm: int = 20,
    ) -> None:
        self._bind_address = bind_address
        self._fps = max(fps, 1.0)
        self._jpeg_quality = int(jpeg_quality)
        self._sndhwm = int(sndhwm)

        self._latest: dict[str, tuple[np.ndarray, float]] = {}
        self._last_published_ts: dict[str, float] = {}
        self._lock = threading.Lock()

        self._ctx: Any = None
        self._socket: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if self._enabled:
            return
        if zmq is None:
            logger.warning(
                "pyzmq not installed; ZmqFramePublisher disabled. "
                "Install with `pip install pyzmq` to enable raw-frame streaming for LeRobot."
            )
            return
        if cv2 is None:
            logger.warning("cv2 not available; ZmqFramePublisher disabled.")
            return

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self._sndhwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind_address)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._publish_loop,
            name="zmq-frame-publisher",
            daemon=True,
        )
        self._thread.start()
        self._enabled = True
        logger.info(
            "ZmqFramePublisher bound at %s (fps=%.1f, jpeg_quality=%d)",
            self._bind_address,
            self._fps,
            self._jpeg_quality,
        )

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if self._socket is not None:
                self._socket.close(linger=0)
        finally:
            self._socket = None
        self._enabled = False

    def publish(self, camera_id: str, frame_rgb: np.ndarray) -> None:
        """Hand a fresh RGB frame to the publisher. Non-blocking.

        The frame reference is stored under a lock and overwrites any prior
        unsent frame for the same camera. The caller must not mutate the array
        after handing it off (live_camera.py creates a fresh array each loop).
        """
        if not self._enabled:
            return
        ts = time.time()
        with self._lock:
            self._latest[camera_id] = (frame_rgb, ts)

    def _publish_loop(self) -> None:
        assert zmq is not None  # _enabled guards this
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            tick = time.perf_counter()

            with self._lock:
                pending = {
                    name: (frame, ts)
                    for name, (frame, ts) in self._latest.items()
                    if ts > self._last_published_ts.get(name, 0.0)
                }

            if pending:
                images: dict[str, str] = {}
                timestamps: dict[str, float] = {}
                for name, (frame, ts) in pending.items():
                    encoded = self._encode_jpeg(frame)
                    if encoded is None:
                        continue
                    images[name] = encoded
                    timestamps[name] = ts
                    self._last_published_ts[name] = ts

                if images:
                    payload = json.dumps({"timestamps": timestamps, "images": images})
                    try:
                        self._socket.send_string(payload, zmq.NOBLOCK)
                    except zmq.Again:
                        logger.debug("ZMQ send dropped: HWM reached")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ZMQ send error: %s", exc)

            elapsed = time.perf_counter() - tick
            remaining = interval - elapsed
            if remaining > 0:
                self._stop.wait(remaining)

    def _encode_jpeg(self, frame_rgb: np.ndarray) -> str | None:
        assert cv2 is not None
        try:
            ok, buffer = cv2.imencode(
                ".jpg",
                frame_rgb,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("JPEG encode error: %s", exc)
            return None
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")
