from __future__ import annotations

import asyncio
from collections import deque
from threading import Lock
from typing import Any


class SceneBus:
    """In-memory latest-scene bus for local development adapters."""

    def __init__(self) -> None:
        self._latest_scene: dict[str, Any] | None = None
        self._latest_by_camera: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=256)
        self._sequence = 0
        self._latest_lock = Lock()
        self._condition: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._condition = asyncio.Condition()

    def publish_nowait(self, scene: dict[str, Any]) -> None:
        with self._latest_lock:
            self._sequence += 1
            scene_copy = dict(scene)
            scene_copy["_bus_sequence"] = self._sequence
            self._latest_scene = scene_copy
            self._events.append(scene_copy)
            camera_id = scene_copy.get("camera_id")
            if isinstance(camera_id, str) and camera_id:
                self._latest_by_camera[camera_id] = scene_copy

        if self._loop is None or self._condition is None:
            return

        async def _notify() -> None:
            assert self._condition is not None
            async with self._condition:
                self._condition.notify_all()

        asyncio.run_coroutine_threadsafe(_notify(), self._loop)

    async def wait_for_next(
        self,
        last_key: tuple[Any, Any, Any, Any] | None = None,
        *,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        if self._condition is None:
            raise RuntimeError("SceneBus loop is not attached")

        last_sequence = self._last_sequence(last_key)

        def _has_new_scene() -> bool:
            with self._latest_lock:
                return self._next_scene_after(last_sequence, camera_id=camera_id) is not None

        async with self._condition:
            await self._condition.wait_for(_has_new_scene)

        with self._latest_lock:
            scene = self._next_scene_after(last_sequence, camera_id=camera_id)
            return dict(scene or {})

    def latest(self, camera_id: str | None = None) -> dict[str, Any] | None:
        with self._latest_lock:
            scene = self._latest_by_camera.get(camera_id) if camera_id is not None else self._latest_scene
            if scene is None:
                return None
            return self._public_scene(scene)

    @staticmethod
    def _public_scene(scene: dict[str, Any]) -> dict[str, Any]:
        scene_copy = dict(scene)
        scene_copy.pop("_bus_sequence", None)
        return scene_copy

    @staticmethod
    def _last_sequence(last_key: tuple[Any, Any, Any, Any] | None) -> int | None:
        if last_key is None:
            return None
        sequence = last_key[0]
        return sequence if isinstance(sequence, int) else None

    def _next_scene_after(self, last_sequence: int | None, *, camera_id: str | None) -> dict[str, Any] | None:
        if last_sequence is None:
            return self._latest_by_camera.get(camera_id) if camera_id is not None else self._latest_scene

        for scene in self._events:
            sequence = scene.get("_bus_sequence")
            if not isinstance(sequence, int) or sequence <= last_sequence:
                continue
            if camera_id is not None and scene.get("camera_id") != camera_id:
                continue
            return scene
        return None


scene_bus = SceneBus()
