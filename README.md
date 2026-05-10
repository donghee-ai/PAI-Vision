# PAI-Vision

Physical AI 파이프라인용 **Vision 모듈 실험 레포**입니다.

핵심 역할:
- 카메라 입력 받기
- YOLO 기반 객체 인식 수행
- 후속 모듈이 읽기 쉬운 **scene JSON** 내보내기

장기적으로 hub / ROS2 bridge / orchestration 책임은 이 레포 밖에서 담당하고,
이 레포의 HTTP/WebSocket은 **로컬 개발용 adapter**로 유지합니다.

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

CAMERA_TARGET_FPS=10
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
SCENE_JSON_PATH=runtime/latest_scene_{camera_id}.json
SCENE_LOG_DIR=runtime/logs
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

## 다음 단계

- `track_id` 안정화
- external hub / ROS2 bridge 연결
- 커스텀 tabletop 데이터셋 fine-tuning
