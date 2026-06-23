# PAI-Vision

Physical AI 파이프라인의 **Vision 모듈**.

- USB 카메라 단일 소유 → **YOLOE 오픈 보캐뷸러리 인식** → **scene JSON** (WebSocket/HTTP)
- 동일 카메라의 **raw 프레임을 ZMQ로 송출** → LeRobot 정책이 `ZMQCamera`로 그대로 소비
- 인식된 물체만 남기고 **배경을 제거한 누끼(전경 컷아웃)를 WebSocket/ZMQ로 송출**

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
            ├──► YOLOE → WebSocket /ws/scenes (scene JSON)
            │
            └──► 마스크 합집합 → 누끼(RGBA) ─┐
                                            ▼
                          Cutout Streamer (CUTOUT_FPS, 별도 스레드)
                            ├──► WebSocket /ws/cutouts
                            └──► ZMQ PUB :5556 (CUTOUT_ZMQ_ENABLED)
```

캡처·YOLO·인코딩이 각각 다른 스레드라, **YOLO가 느려도 ZMQ raw 송출과 누끼 인코딩 rate는 떨어지지 않습니다.**

> **모델**: 기본 `yoloe-11s-seg.pt`(오픈 보캐뷸러리 + 세그멘테이션). `YOLO_CLASSES`로 인식 어휘를
> 자유롭게 지정합니다(미지정 시 아무것도 탐지하지 않음). 닫힌 어휘 모델(`yolo11s-seg.pt`)도 그대로
> 동작하며 이때 `YOLO_CLASSES`는 무시됩니다. YOLOE 텍스트 프롬프트는 최초 1회 CLIP 패키지를
> 자동 설치합니다(인터넷 필요).

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
YOLO_MODEL=yoloe-11s-seg.pt
YOLO_DEVICE=auto
# 오픈 보캐뷸러리 프롬프트 (YOLOE/YOLO-World 전용). 미지정 시 YOLOE는 아무것도 탐지하지 않음.
YOLO_CLASSES=person,orange,plate,cup,bottle,bowl,banana,apple,box,scissors

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
- scene 스트림: `ws://localhost:8000/ws/scenes?camera_id=front_rgb`
- scene 뷰어: `http://localhost:8000/viewer`
- **누끼 스트림**: `ws://localhost:8000/ws/cutouts?camera_id=front_rgb` (base64 이미지 프레임)
- **누끼 뷰어**: `http://localhost:8000/cutout/viewer` (브라우저 미리보기, 투명 배경)
- **최신 누끼**: `GET http://localhost:8000/cutout/latest?camera_id=front_rgb` (디코드된 이미지)
- 단일 이미지 누끼: `POST http://localhost:8000/predict/cutout` (파일 업로드 → 투명 PNG)

### 누끼(전경 컷아웃) 스트리밍

`CUTOUT_ENABLED=true`(기본)면 인식된 물체의 마스크 합집합으로 배경을 투명 처리한 RGBA 프레임을
`/ws/cutouts`로 송출합니다. ZMQ로도 보내려면 `CUTOUT_ZMQ_ENABLED=true`(별도 포트 `:5556`, raw 프레임
`:5555`와 분리). 형식은 `CUTOUT_FORMAT=webp|png`. 인코딩은 전용 스레드에서 돌아 YOLO 루프에 부담을
주지 않습니다.

ZMQ 누끼 와이어 포맷(틱당 1메시지, 카메라 배치):

```json
{ "image_format": "webp",
  "timestamps": {"front_rgb": 1234.5},
  "images":     {"front_rgb": "<base64 webp/png, alpha 포함>"} }
```

---

## LeRobot 연동

PAI-Vision에서 `ZMQ_PUBLISH_ENABLED=true`로 띄운 뒤, LeRobot 명령에 ZMQ 카메라 지정:

```bash
--robot.cameras="{front_rgb: {type: zmq, server_address: localhost, port: 5555, camera_name: front_rgb, width: 1280, height: 720, fps: 30}}"
```

`camera_name`은 PAI-Vision의 `CAMERA_ID`와 정확히 같아야 합니다. **수집/추론 모두 동일 파이프라인 유지 필수** (입력 분포 일관성).
