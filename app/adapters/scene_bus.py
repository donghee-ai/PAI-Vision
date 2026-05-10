from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any


class SceneBus:
    """In-memory latest-scene bus for local development adapters."""

    def __init__(self) -> None:
        self._latest_scene: dict[str, Any] | None = None
        self._latest_lock = Lock()
        self._condition: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._condition = asyncio.Condition()

    def publish_nowait(self, scene: dict[str, Any]) -> None:
        with self._latest_lock:
            self._latest_scene = dict(scene)

        if self._loop is None or self._condition is None:
            return

        async def _notify() -> None:
            assert self._condition is not None
            async with self._condition:
                self._condition.notify_all()

        asyncio.run_coroutine_threadsafe(_notify(), self._loop)

    async def wait_for_next(self, last_key: tuple[Any, Any, Any] | None = None) -> dict[str, Any]:
        if self._condition is None:
            raise RuntimeError("SceneBus loop is not attached")

        def _has_new_scene() -> bool:
            with self._latest_lock:
                return self._latest_scene is not None and self._scene_key(self._latest_scene) != last_key

        async with self._condition:
            await self._condition.wait_for(_has_new_scene)

        with self._latest_lock:
            return dict(self._latest_scene or {})

    def latest(self) -> dict[str, Any] | None:
        with self._latest_lock:
            if self._latest_scene is None:
                return None
            return dict(self._latest_scene)

    @staticmethod
    def _scene_key(scene: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (
            scene.get("frame_id"),
            scene.get("timestamp"),
            scene.get("camera_id"),
        )


scene_bus = SceneBus()
