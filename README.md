# PAI-Vision

Physical AI 파이프라인의 **Vision 모듈**.

- USB 카메라 단일 소유 → YOLO 객체 인식 → **scene JSON** (WebSocket/HTTP)
- 동일 카메라의 **raw 프레임을 ZMQ로 송출** → LeRobot 정책이 `ZMQCamera`로 그대로 소비

### 아키텍처

```
USB 카메라
   │
   ▼
Capture Thread (CAMERA_CAPTURE_FPS)
   │
   ├──► ZMQ PUB :5555  ──► LeRobot (raw 프레임)
   │
   └──► FrameBuffer (latest wins)
            │
            ▼
       Perception Thread (CAMERA_TARGET_FPS)
            │
            └──► YOLO → WebSocket /ws/scenes (PAI-Language)
```

캡처와 YOLO가 분리되어 있어, **YOLO가 느려도 ZMQ 송출 rate는 떨어지지 않습니다.**

---

## 세팅

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env
```

### 환경변수 (`.env`)

```env
YOLO_MODEL=yolo11s-seg.pt
YOLO_DEVICE=auto

# 단일 카메라
CAMERA_ID=front_rgb
CAMERA_INDEX=0

# 또는 멀티 카메라 (지정 시 우선)
# CAMERAS=front_rgb:0,side_rgb:1

CAMERA_CAPTURE_FPS=30   # 캡처 + ZMQ 송출
CAMERA_TARGET_FPS=10    # YOLO 추론 (느려도 ZMQ 영향 X)
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# LeRobot 연동 시 활성화
ZMQ_PUBLISH_ENABLED=false
ZMQ_PUBLISH_BIND=tcp://*:5555
```

---

## 시작

### 카메라 + adapter 함께 (기본 사용법)
```bash
python -m app.adapters.run_all
```

### 멀티 카메라
```bash
python -m app.adapters.run_all --cameras front_rgb:0,side_rgb:1
```

### 카메라 없이 adapter만
```bash
uvicorn app.adapters.local_api:app --host 0.0.0.0 --port 8000
```

### 엔드포인트
- 최신 scene: `GET http://localhost:8000/scene/latest?camera_id=front_rgb`
- 스트림: `ws://localhost:8000/ws/scenes?camera_id=front_rgb`
- 뷰어: `http://localhost:8000/viewer`

---

## LeRobot 연동

PAI-Vision에서 `ZMQ_PUBLISH_ENABLED=true`로 띄운 뒤, LeRobot 명령에 ZMQ 카메라 지정:

```bash
--robot.cameras="{front_rgb: {type: zmq, server_address: localhost, port: 5555, camera_name: front_rgb, width: 1280, height: 720, fps: 30}}"
```

`camera_name`은 PAI-Vision의 `CAMERA_ID`와 정확히 같아야 합니다. **수집/추론 모두 동일 파이프라인 유지 필수** (입력 분포 일관성).
