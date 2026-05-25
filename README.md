# PAI-Vision

Physical AI 파이프라인용 **Vision 모듈 실험 레포**입니다.

핵심 역할:
- 카메라 입력 받기 (USB 카메라 단일 소유)
- YOLO 기반 객체 인식 수행
- 후속 모듈이 읽기 쉬운 **scene JSON** 내보내기 (WebSocket / HTTP)
- **raw 프레임 ZMQ 송출** — LeRobot 정책이 동일 카메라 스트림을 소비

장기적으로 hub / ROS2 bridge / orchestration 책임은 이 레포 밖에서 담당하고,
이 레포의 HTTP/WebSocket/ZMQ는 **로컬 개발용 adapter**로 유지합니다.

### 스트리밍 아키텍처

```
USB 카메라
   │
   ▼
Capture Thread (CAMERA_CAPTURE_FPS, 기본 30Hz)
   │
   ├──► on_frame ──► ZMQ PUB :5555  ──► LeRobot ZMQCamera (raw 프레임)
   │
   └──► FrameBuffer (단일 슬롯, latest wins)
            │
            ▼
       Perception Thread (CAMERA_TARGET_FPS, 기본 10Hz)
            │
            └──► YOLO → scene_bus → WebSocket /ws/scenes (PAI-Language용)
```

YOLO가 느려져도 ZMQ 송출 rate는 영향받지 않습니다. 캡처/지각 스레드가 완전히 분리되어 있고, 프레임 버퍼는 항상 최신 1장만 유지합니다 ("latest wins").

---

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env
```

### 단일 카메라 실행
```bash
python -m app.perception.live_camera --no-display --max-frames 10
```

### adapter + 카메라 함께 실행
```bash
python -m app.adapters.run_all --no-display --max-frames 10
```

---

## 환경변수

주요 항목:

```env
YOLO_MODEL=yolo11s-seg.pt
YOLO_DEVICE=auto
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_IOU=0.7

CAMERA_ID=front_rgb
CAMERA_INDEX=0
CAMERAS=

# Capture / perception 분리된 FPS 설정
CAMERA_CAPTURE_FPS=30   # 캡처 + ZMQ 송출 rate (LeRobot 학습 FPS와 일치 권장)
CAMERA_TARGET_FPS=10    # YOLO 추론 rate cap (느려도 ZMQ에 영향 X)

CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
SCENE_JSON_PATH=runtime/latest_scene_{camera_id}.json
SCENE_LOG_DIR=runtime/logs

# ZMQ raw-frame publisher (LeRobot 연동)
ZMQ_PUBLISH_ENABLED=false
ZMQ_PUBLISH_BIND=tcp://*:5555
ZMQ_PUBLISH_FPS=30
ZMQ_PUBLISH_JPEG_QUALITY=90
```

### 단일 카메라
- `CAMERA_ID`, `CAMERA_INDEX` 사용

예:
```env
CAMERA_ID=front_rgb
CAMERA_INDEX=0
```

### 멀티카메라
- `CAMERAS` 사용
- 형식: `camera_id:index,camera_id:index`

예:
```env
CAMERAS=front_rgb:0,side_rgb:1,top_rgb:2
```

`CAMERAS`가 있으면 멀티카메라 worker가 우선됩니다.

---

## 멀티카메라 실행 예시

```bash
python -m app.adapters.run_all --cameras front_rgb:0,side_rgb:1 --no-display
```

또는 `.env`에:

```env
CAMERAS=front_rgb:0,side_rgb:1
```

그 후:

```bash
python -m app.adapters.run_all --no-display
```

---

## 출력

### scene 파일
카메라별로 scene 파일이 분리됩니다.

예:
- `runtime/latest_scene_front_rgb.json`
- `runtime/latest_scene_side_rgb.json`

### session log
카메라별 로그 파일이 생성됩니다.

예:
- `runtime/logs/live_camera_front_rgb_YYYYMMDD_HHMMSS.jsonl`
- `runtime/logs/live_camera_side_rgb_YYYYMMDD_HHMMSS.jsonl`

---

## Local API / WebSocket adapter

### 서버 실행
```bash
uvicorn app.adapters.local_api:app --host 0.0.0.0 --port 8000 --reload
```

### latest scene 조회
전체 최신 scene:
```bash
curl "http://localhost:8000/scene/latest"
```

특정 카메라 최신 scene:
```bash
curl "http://localhost:8000/scene/latest?camera_id=front_rgb"
```

### websocket stream
전체 카메라 update stream:
```text
ws://localhost:8000/ws/scenes
```

특정 카메라만:
```text
ws://localhost:8000/ws/scenes?camera_id=front_rgb&max_fps=10
```

### viewer
```text
http://localhost:8000/viewer
```

---

## scene JSON 메모

주요 필드:
- `camera_id`
- `frame_id`
- `timestamp`
- `objects[]`
- `bbox_xyxy`
- `center_pixel`
- `status`

이 레포는 현재 **2D RGB camera scene understanding** 기준입니다.
깊이/3D 좌표는 아직 포함하지 않습니다.

---

## 현재 구조

- `app/perception`
  - 추론, tracking, scene 생성
- `app/adapters`
  - local API, websocket, run_all

즉:
- **perception = 보기 / 인식하기**
- **adapter = 개발 중 연결/관찰하기**

---

## LeRobot 연동 (ZMQ raw-frame)

PAI-Vision이 카메라를 단일 소유하면서 raw 프레임을 ZMQ로 송출하면, LeRobot 정책은
`ZMQCamera` 백엔드로 동일 스트림을 소비할 수 있습니다 (USB 카메라 점유 충돌 없음).

### 1. PAI-Vision에서 ZMQ 활성화

`.env`:
```env
ZMQ_PUBLISH_ENABLED=true
ZMQ_PUBLISH_BIND=tcp://*:5555
ZMQ_PUBLISH_FPS=30
CAMERA_CAPTURE_FPS=30
```

기동:
```bash
python -m app.adapters.run_all --camera 0 --camera-id front_rgb
```
콘솔에 `ZMQ raw-frame publisher active at tcp://*:5555 ...`가 보이면 OK.

### 2. LeRobot 쪽에서 ZMQCamera로 수신

기록 / 추론 명령 어디서나 (예: `lerobot-record`, `lerobot-rollout`):
```bash
--robot.cameras="{front_rgb: {type: zmq, server_address: localhost, port: 5555, camera_name: front_rgb, width: 1280, height: 720, fps: 30}}"
```

**중요**: LeRobot 쪽 `camera_name`이 PAI-Vision `CAMERA_ID`와 정확히 일치해야 함.

### 3. SUB 한 줄로 sanity check

```python
import zmq, json
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.setsockopt_string(zmq.SUBSCRIBE, ""); s.connect("tcp://localhost:5555")
msg = json.loads(s.recv_string())
print("cameras:", list(msg["images"].keys()), "ts:", msg["timestamps"])
```

학습 데이터 수집과 추론 모두 이 ZMQ 파이프라인으로 통일하면 도메인 일관성이 보장됩니다 (입력 분포가 같음).

---

## 다음 단계

- `track_id` 안정화
- external hub / ROS2 bridge 연결
- 커스텀 tabletop 데이터셋 fine-tuning
