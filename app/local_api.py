"""Local HTTP/WebSocket adapter for PAI-Vision perception outputs.

This module is intentionally scoped to development and integration testing. The
perception pipeline remains responsible for producing scene data; durable hub,
bridge, and orchestration responsibilities should be implemented outside this
repository.
"""

import asyncio
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.device import resolve_yolo_device
from app.scene import build_scene_response
from app.schemas import PredictionResponse, SceneResponse
from app.vision import YoloSegmentationService, render_prediction_overlay

app = FastAPI(title="PAI-Vision Local Perception Adapter", version="0.1.0")
SCENE_STREAM_POLL_SECONDS = 0.05
WS_TOPIC_VISION_UPDATE = "vision_update"
WS_SENDER_VISION = "vision"
SCENE_VIEWER_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PAI Vision Scene Viewer</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f7f8fb;
        color: #16181d;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        background: #f7f8fb;
      }
      main {
        width: min(1180px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 24px 0 32px;
      }
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
      }
      h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 700;
      }
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        min-height: 34px;
        padding: 0 12px;
        border: 1px solid #d7dce5;
        border-radius: 6px;
        background: #ffffff;
        font-size: 14px;
      }
      .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #8b95a5;
      }
      .status.connected .dot {
        background: #12a150;
      }
      .status.error .dot,
      .status.closed .dot {
        background: #d92d20;
      }
      .metrics {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 16px;
      }
      .metric {
        min-height: 76px;
        padding: 12px;
        border: 1px solid #dfe3ea;
        border-radius: 8px;
        background: #ffffff;
      }
      .metric span {
        display: block;
        color: #667085;
        font-size: 12px;
        margin-bottom: 8px;
      }
      .metric strong {
        display: block;
        overflow-wrap: anywhere;
        font-size: 20px;
        line-height: 1.2;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
        gap: 16px;
      }
      section {
        min-width: 0;
      }
      h2 {
        margin: 0 0 10px;
        font-size: 16px;
      }
      .table-wrap,
      pre {
        border: 1px solid #dfe3ea;
        border-radius: 8px;
        background: #ffffff;
      }
      .table-wrap {
        overflow: auto;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        min-width: 720px;
      }
      th,
      td {
        padding: 10px 12px;
        border-bottom: 1px solid #eef1f5;
        text-align: left;
        white-space: nowrap;
        font-size: 13px;
      }
      th {
        color: #475467;
        background: #f2f4f7;
        font-weight: 650;
      }
      tr:last-child td {
        border-bottom: 0;
      }
      pre {
        min-height: 420px;
        max-height: calc(100vh - 214px);
        margin: 0;
        padding: 14px;
        overflow: auto;
        font-size: 12px;
        line-height: 1.45;
      }
      .empty {
        padding: 24px;
        color: #667085;
        font-size: 14px;
      }
      @media (max-width: 860px) {
        main {
          width: min(100vw - 20px, 720px);
          padding-top: 16px;
        }
        header {
          align-items: flex-start;
          flex-direction: column;
        }
        .metrics {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .layout {
          grid-template-columns: 1fr;
        }
        pre {
          max-height: 520px;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>PAI Vision Scene Viewer</h1>
        <div id="status" class="status">
          <span class="dot"></span>
          <span id="statusText">connecting</span>
        </div>
      </header>

      <div class="metrics">
        <div class="metric"><span>Camera</span><strong id="cameraId">-</strong></div>
        <div class="metric"><span>Frame</span><strong id="frameId">-</strong></div>
        <div class="metric"><span>Loop FPS</span><strong id="loopFps">-</strong></div>
        <div class="metric"><span>Objects</span><strong id="objectCount">-</strong></div>
        <div class="metric"><span>Updated</span><strong id="updatedAt">-</strong></div>
      </div>

      <div class="layout">
        <section>
          <h2>Objects</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Track</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Center</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody id="objectRows">
                <tr><td colspan="6" class="empty">Waiting for scene data</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2>Raw Scene JSON</h2>
          <pre id="rawJson">{}</pre>
        </section>
      </div>
    </main>

    <script>
      const params = new URLSearchParams(window.location.search);
      const cameraId = params.get("camera_id") || "front_rgb";
      const maxFps = params.get("max_fps") || "10";
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socketUrl = `${protocol}://${window.location.host}/ws/scenes?camera_id=${encodeURIComponent(cameraId)}&max_fps=${encodeURIComponent(maxFps)}`;
      const status = document.getElementById("status");
      const statusText = document.getElementById("statusText");

      function setStatus(value) {
        status.className = `status ${value}`;
        statusText.textContent = value;
      }

      function valueOrDash(value) {
        return value === null || value === undefined || value === "" ? "-" : value;
      }

      function escapeHtml(value) {
        return String(valueOrDash(value))
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function unwrapSceneMessage(message) {
        if (message && message.type === "vision_update" && message.data) {
          return message.data;
        }
        return message;
      }

      function renderScene(scene) {
        document.getElementById("cameraId").textContent = valueOrDash(scene.camera_id);
        document.getElementById("frameId").textContent = valueOrDash(scene.frame_id);
        document.getElementById("loopFps").textContent = scene.loop_fps === null || scene.loop_fps === undefined ? "-" : Number(scene.loop_fps).toFixed(2);
        document.getElementById("objectCount").textContent = Array.isArray(scene.objects) ? scene.objects.length : 0;
        document.getElementById("updatedAt").textContent = new Date().toLocaleTimeString();
        document.getElementById("rawJson").textContent = JSON.stringify(scene, null, 2);

        const rows = document.getElementById("objectRows");
        const objects = Array.isArray(scene.objects) ? scene.objects : [];
        if (objects.length === 0) {
          rows.innerHTML = '<tr><td colspan="6" class="empty">No objects detected</td></tr>';
          return;
        }

        rows.innerHTML = objects.map((object) => {
          const center = Array.isArray(object.center_pixel) ? `[${object.center_pixel.join(", ")}]` : "-";
          const confidence = object.confidence === null || object.confidence === undefined ? "-" : Number(object.confidence).toFixed(4);
          return `
            <tr>
              <td>${escapeHtml(object.id)}</td>
              <td>${escapeHtml(object.track_id)}</td>
              <td>${escapeHtml(object.label)}</td>
              <td>${confidence}</td>
              <td>${escapeHtml(center)}</td>
              <td>${escapeHtml(object.status)}</td>
            </tr>
          `;
        }).join("");
      }

      setStatus("connecting");
      const ws = new WebSocket(socketUrl);
      ws.onopen = () => setStatus("connected");
      ws.onmessage = (event) => renderScene(unwrapSceneMessage(JSON.parse(event.data)));
      ws.onerror = () => setStatus("error");
      ws.onclose = () => setStatus("closed");
    </script>
  </body>
</html>
"""


@lru_cache
def get_service() -> YoloSegmentationService:
    settings = get_settings()
    return YoloSegmentationService(
        model_path=settings.yolo_model,
        device=settings.yolo_device,
    )


@app.get("/health")
def health() -> dict[str, str | int | float]:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.yolo_model,
        "device": settings.yolo_device,
        "resolved_device": resolve_yolo_device(settings.yolo_device),
        "imgsz": settings.yolo_imgsz,
        "conf": settings.yolo_conf,
        "iou": settings.yolo_iou,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> PredictionResponse:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )


@app.post("/predict/scene", response_model=SceneResponse)
async def predict_scene(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> SceneResponse:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    started = perf_counter()
    prediction = get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )
    inference_ms = (perf_counter() - started) * 1000
    return build_scene_response(
        prediction,
        frame_id=0,
        camera_id=settings.camera_id,
        inference_ms=inference_ms,
    )


@app.post("/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=2048),
) -> Response:
    settings = get_settings()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    prediction = get_service().predict(
        image,
        conf=settings.yolo_conf if conf is None else conf,
        iou=settings.yolo_iou if iou is None else iou,
        imgsz=settings.yolo_imgsz if imgsz is None else imgsz,
    )
    rendered = render_prediction_overlay(image, prediction)
    output = BytesIO()
    rendered.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")


@app.get("/scene/latest")
def latest_scene() -> Response:
    settings = get_settings()
    scene_path = Path(settings.scene_json_path)
    if not scene_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene JSON not found: {scene_path}")
    return Response(content=scene_path.read_text(encoding="utf-8"), media_type="application/json")


@app.get("/viewer", response_class=HTMLResponse)
def scene_viewer() -> HTMLResponse:
    return HTMLResponse(SCENE_VIEWER_HTML)


@app.websocket("/ws/scenes")
async def scene_stream(
    websocket: WebSocket,
    camera_id: str | None = None,
    max_fps: float = 10.0,
) -> None:
    """Development-only stream of the local scene JSON file.

    This keeps the current browser/debug workflow available. It is intentionally
    a thin adapter around perception output, not the long-term orchestration hub
    or ROS2 bridge.
    """
    await websocket.accept()
    settings = get_settings()
    scene_path = Path(settings.scene_json_path)
    min_send_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_sent_key: tuple[object, object, object] | None = None
    last_send_time = 0.0

    try:
        while True:
            scene = _read_latest_scene(scene_path)
            if scene is None:
                await asyncio.sleep(SCENE_STREAM_POLL_SECONDS)
                if not await _websocket_is_connected(websocket):
                    return
                continue

            if camera_id is not None and scene.get("camera_id") != camera_id:
                await asyncio.sleep(SCENE_STREAM_POLL_SECONDS)
                if not await _websocket_is_connected(websocket):
                    return
                continue

            scene_key = (
                scene.get("frame_id"),
                scene.get("timestamp"),
                scene.get("camera_id"),
            )
            now = perf_counter()
            if scene_key != last_sent_key and now - last_send_time >= min_send_interval:
                await websocket.send_json(_build_scene_envelope(scene))
                last_sent_key = scene_key
                last_send_time = now

            await asyncio.sleep(SCENE_STREAM_POLL_SECONDS)
            if not await _websocket_is_connected(websocket):
                return
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return


def _read_latest_scene(scene_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(scene_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _build_scene_envelope(scene: dict[str, object]) -> dict[str, object]:
    return {
        "type": WS_TOPIC_VISION_UPDATE,
        "timestamp": scene.get("timestamp"),
        "sender": WS_SENDER_VISION,
        "data": scene,
    }


async def _websocket_is_connected(websocket: WebSocket) -> bool:
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout=0.001)
    except asyncio.TimeoutError:
        return True
    except WebSocketDisconnect:
        return False

    return message.get("type") != "websocket.disconnect"
