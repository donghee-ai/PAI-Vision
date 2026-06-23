"""Foreground-cutout streamer: one off-loop encoder, two transports.

The perception loop hands each frame's RGBA cutout (objects opaque, background
transparent) to :meth:`CutoutStreamer.publish`, which is non-blocking. A single
background thread snapshots the latest cutout per camera, encodes it once to
base64 WEBP/PNG, and fans the result out to:

- the in-memory ``cutout_bus`` (consumed by the ``/ws/cutouts`` WebSocket), and
- an optional dedicated ZMQ PUB socket (default ``tcp://*:5556``), separate from
  the raw-frame publisher on ``:5555``.

Encoding therefore never charges the YOLO loop budget, and both transports share
a single encode. WEBP is used because, unlike the raw-frame JPEG stream, the
cutout needs an alpha channel for the transparent background.

ZMQ wire format (one message per tick, all cameras batched)::

    {
        "image_format": "webp",
        "timestamps": {"<camera_id>": float, ...},
        "images":     {"<camera_id>": "<base64 webp/png with alpha>", ...}
    }
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from app.adapters.scene_bus import SceneBus

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:  # pyzmq is optional; WebSocket path works without it
    zmq = None  # type: ignore[assignment]

# Consumed by the FastAPI /ws/cutouts endpoint. run_all attaches the event loop.
cutout_bus = SceneBus()

WS_TOPIC_CUTOUT_UPDATE = "cutout_update"


class CutoutStreamer:
    """Background encoder that streams foreground cutouts over WS (+ optional ZMQ).

    Thread-safe: :meth:`publish` may be called concurrently from multiple camera
    perception threads (one per camera_id).
    """

    def __init__(
        self,
        *,
        fps: float = 15.0,
        image_format: str = "webp",
        quality: int = 80,
        zmq_enabled: bool = False,
        zmq_bind: str = "tcp://*:5556",
        sndhwm: int = 10,
    ) -> None:
        self._fps = max(fps, 1.0)
        self._format = image_format.strip().lower()
        if self._format not in {"webp", "png"}:
            logger.warning("Unknown cutout format %r; falling back to webp", image_format)
            self._format = "webp"
        self._quality = int(quality)
        self._zmq_enabled = bool(zmq_enabled)
        self._zmq_bind = zmq_bind
        self._sndhwm = int(sndhwm)

        self._latest: dict[str, tuple[np.ndarray, dict[str, Any], float]] = {}
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
        if self._zmq_enabled and zmq is None:
            logger.warning(
                "pyzmq not installed; cutout ZMQ publishing disabled "
                "(the /ws/cutouts WebSocket still works)."
            )
            self._zmq_enabled = False
        if self._zmq_enabled:
            self._ctx = zmq.Context.instance()
            self._socket = self._ctx.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, self._sndhwm)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(self._zmq_bind)

        self._stop.clear()
        self._thread = threading.Thread(target=self._encode_loop, name="cutout-streamer", daemon=True)
        self._thread.start()
        self._enabled = True
        logger.info(
            "CutoutStreamer started (fps=%.1f, format=%s, zmq=%s)",
            self._fps,
            self._format,
            self._zmq_bind if self._zmq_enabled else "off",
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

    def publish(self, camera_id: str, cutout_rgba: np.ndarray, meta: dict[str, Any]) -> None:
        """Hand a fresh RGBA cutout to the streamer. Non-blocking, latest-wins.

        The caller must not mutate ``cutout_rgba`` afterwards (the perception loop
        allocates a fresh array per frame via ``build_foreground_cutout_rgba``).
        """
        if not self._enabled:
            return
        ts = time.time()
        with self._lock:
            self._latest[camera_id] = (cutout_rgba, dict(meta), ts)

    def _encode_loop(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            tick = time.perf_counter()

            with self._lock:
                pending = {
                    name: (frame, meta, ts)
                    for name, (frame, meta, ts) in self._latest.items()
                    if ts > self._last_published_ts.get(name, 0.0)
                }

            if pending:
                images: dict[str, str] = {}
                timestamps: dict[str, float] = {}
                for name, (frame, meta, ts) in pending.items():
                    encoded = self._encode(frame)
                    if encoded is None:
                        continue
                    images[name] = encoded
                    timestamps[name] = ts
                    self._last_published_ts[name] = ts
                    cutout_bus.publish_nowait(
                        {
                            "type": WS_TOPIC_CUTOUT_UPDATE,
                            "camera_id": name,
                            "frame_id": meta.get("frame_id"),
                            "timestamp": meta.get("timestamp"),
                            "image_format": self._format,
                            "image_size": meta.get("image_size"),
                            "object_count": meta.get("object_count"),
                            "image": encoded,
                        }
                    )

                if images and self._zmq_enabled and self._socket is not None:
                    payload = json.dumps(
                        {"image_format": self._format, "timestamps": timestamps, "images": images}
                    )
                    try:
                        self._socket.send_string(payload, zmq.NOBLOCK)
                    except zmq.Again:
                        logger.debug("cutout ZMQ send dropped: HWM reached")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cutout ZMQ send error: %s", exc)

            elapsed = time.perf_counter() - tick
            remaining = interval - elapsed
            if remaining > 0:
                self._stop.wait(remaining)

    def _encode(self, cutout_rgba: np.ndarray) -> str | None:
        try:
            image = Image.fromarray(cutout_rgba, mode="RGBA")
            buffer = io.BytesIO()
            if self._format == "png":
                image.save(buffer, format="PNG")
            else:
                image.save(buffer, format="WEBP", quality=self._quality)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cutout encode error: %s", exc)
            return None
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
