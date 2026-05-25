"""Single-slot latest-frame buffer for decoupling capture from perception.

Producer (capture thread) calls `put(frame_rgb)` at the camera's native rate.
Consumer (perception thread) calls `get(timeout=...)` and always receives the
*latest* available frame, never a queued older one. Frames produced while the
consumer is busy are silently overwritten — "latest wins" semantics.

This pattern keeps ZMQ raw-frame publishing independent from YOLO inference
latency: capture and ZMQ-out run at full speed in their own thread, while
perception runs at its own (typically slower) cadence.
"""

from __future__ import annotations

import threading

import numpy as np


class FrameBuffer:
    """Thread-safe one-slot latest-frame buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._unread_id = 0

    def put(self, frame_rgb: np.ndarray) -> None:
        """Store the latest frame, overwriting any unread predecessor.

        The caller must not mutate the array after handing it off. live_camera's
        capture loop satisfies this because cv2.cvtColor allocates a fresh array
        on every call.
        """
        with self._lock:
            self._frame = frame_rgb
            self._frame_id += 1
            self._unread_id = self._frame_id
        self._new_frame.set()

    def get(self, timeout: float | None = None) -> tuple[np.ndarray | None, int]:
        """Block until a new (unread) frame is available, then return it.

        Returns (None, 0) on timeout. Consecutive calls without a new put() will
        block — the buffer never returns the same frame twice.
        """
        if not self._new_frame.wait(timeout):
            return None, 0
        with self._lock:
            frame = self._frame
            fid = self._unread_id
            self._new_frame.clear()
        return frame, fid

    def clear(self) -> None:
        with self._lock:
            self._frame = None
            self._new_frame.clear()
