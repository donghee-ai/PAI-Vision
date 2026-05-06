# PAI-Vision

Physical AI 기반 VLA 파이프라인의 Vision 모듈 실험 레포입니다.

이 레포의 현재 목표는 **카메라/이미지 입력 → YOLO 기반 물체 인식 → Language/Action 파트가 읽을 수 있는 compact scene JSON 출력** 흐름을 빠르게 검증하는 것입니다.

현재 버전은 pretrained `YOLO11s-seg` 모델로 카메라 프레임을 실시간 추론하고, 인식 결과를 화면에 overlay하며, compact scene JSON과 세션 로그를 갱신합니다.

## Quick Start

### 로컬 실행 최소 절차
1. Python 가상환경 생성
2. 본인 환경에 맞는 PyTorch 설치
3. `requirements.txt` 설치
4. `.env` 생성
5. 카메라 추론 또는 API 서버 실행

예시:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env
python -m app.live_camera --no-display --max-frames 10
```

Apple Silicon 맥북 에어에서는 `YOLO_DEVICE=auto`로 두면 MPS를 우선 사용하고, MPS를 못 쓰면 CPU로 자동 전환합니다.

### 입력 / 출력
- **입력**: 웹캠 프레임 또는 업로드 이미지
- **추론**: YOLO11 segmentation
- **출력**:
  - overlay된 시각화 화면/PNG
  - full prediction JSON
  - Language/Action 파트 전달용 compact scene JSON

## Scene JSON contract

`/predict/scene` 및 `runtime/latest_scene.json`의 목적은 Vision 결과를 후속 모듈이 안정적으로 읽게 하는 것입니다.

현재 기준:
- `center_pixel`은 **입력 이미지 기준 픽셀 좌표**입니다.
- 좌표계 원점은 **좌상단 `(0, 0)`** 입니다.
- `bbox_xyxy`는 `[x1, y1, x2, y2]` 형식입니다.
- `depth_m`, `camera_xyz`, `robot_xyz`는 아직 미구현이라 기본적으로 `null`일 수 있습니다.
- 즉 현재 버전은 **2D scene understanding baseline** 으로 보고, 이후 depth / camera calibration / robot transform 단계로 확장합니다.

예시:

```json
{
  "frame_id": 128,
  "camera_id": "front_rgb",
  "objects": [
    {
      "id": "obj_01",
      "label": "mouse",
      "confidence": 0.94,
      "bbox_xyxy": [331.23, 189.97, 971.52, 1228.74],
      "center_pixel": [655, 700],
      "depth_m": null,
      "camera_xyz": null,
      "robot_xyz": null,
      "status": "detected"
    }
  ]
}
```

## PAI integration intent

이 레포는 단순 detection demo가 아니라, 이후 PAI 파이프라인에서 다음 형태로 연결되는 것을 의도합니다.

- Vision: 물체 후보 추출
- Language: 현재 장면 이해 및 목표 선택
- Action: 선택된 물체에 대한 행동 계획

즉 Vision 모듈의 핵심 책임은 **“무엇이 어디에 있는지”를 후속 모듈이 쓰기 쉬운 형태로 넘기는 것**입니다.

## 1. 환경 전략

이 프로젝트에서는 `torch`, `torchvision`, `torchaudio`를 `requirements.txt`에 넣지 않습니다.

이유는 실행 환경마다 CUDA/PyTorch 조합이 다르기 때문입니다.

| 환경 | 권장 방식 |
| --- | --- |
| MacBook Air / Apple Silicon | `pip install torch torchvision torchaudio` 후 `YOLO_DEVICE=auto` |
| 로컬 RTX 5060 | `.venv` + CUDA 13.0 PyTorch wheel |
| RTX PRO 6000 Blackwell 서버 | 기존 CUDA 13 PyTorch Docker 이미지 |
| 다른 GPU 서버 | Docker base image로 CUDA/PyTorch 고정 |
| Docker를 못 쓰는 서버 | conda 또는 `.venv` 대안 사용 |

`requirements.txt`는 FastAPI, Ultralytics, OpenCV, Pillow 같은 앱 공통 의존성만 관리합니다.

## 2. MacBook Air / Apple Silicon 실행 준비

Apple Silicon Mac에서는 일반적으로 PyTorch 공식 wheel을 설치하면 MPS를 사용할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

MPS 사용 가능 여부를 확인하려면:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
```

예상 예시:

```text
2.7.0
True
True
```

`.env`를 만들고 `YOLO_DEVICE=auto`를 유지합니다.

```bash
cp .env.example .env
```

```env
YOLO_MODEL=yolo11s-seg.pt
YOLO_DEVICE=auto
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_IOU=0.7
CAMERA_ID=front_rgb
CAMERA_INDEX=0
CAMERA_TARGET_FPS=10
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
SCENE_JSON_PATH=runtime/latest_scene.json
SCENE_LOG_DIR=runtime/logs
```

`YOLO_DEVICE`는 `auto`, `cpu`, `mps`, `cuda:0`, `cuda:1` 같은 값을 받을 수 있습니다.
`auto`는 현재 실행 가능한 디바이스를 `CUDA -> MPS -> CPU` 순서로 고릅니다.

## 3. 로컬 RTX 5060 실행 준비

Windows PowerShell 기준입니다.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

GPU 인식 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NO CUDA')"
```

정상 예시:

```text
2.10.0+cu130
13.0
True
NVIDIA GeForce RTX 5060
(12, 0)
```

`.env`를 만들고 GPU device를 지정합니다.

```bash
Copy-Item .env.example .env
```

`YOLO_DEVICE=cuda:0`처럼 명시적으로 특정 GPU를 지정할 수 있습니다.

## 4. CPU Torch가 깔렸을 때 복구

아래처럼 나오면 CPU 전용 PyTorch가 설치된 상태입니다.

```text
2.11.0+cpu
None
False
Torch not compiled with CUDA enabled
```

이 경우 지우고 CUDA 13.0 wheel로 다시 설치합니다.

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

그 다음 다시 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 5. RTX PRO 6000 Blackwell 서버

서버에서는 이미 CUDA 13 PyTorch Docker 이미지를 사용합니다.

예시:

```text
pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel
nvcr.io/nvidia/pytorch:25.12-py3
```

이 경우 컨테이너 안에서 `torch`를 다시 설치하지 않습니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7071
```

서버에서 특정 GPU만 사용하려면:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn app.main:app --host 0.0.0.0 --port 7071
```

Docker 이미지가 이미 CUDA/PyTorch 버전을 고정하므로, 이 환경에서는 conda를 추가로 쓰지 않는 편이 단순합니다.

## 6. 카메라 실시간 추론

웹캠을 열고 10 FPS 목표로 YOLO11-Seg 결과를 화면에 띄웁니다.

```bash
python -m app.live_camera
```

명시적으로 옵션을 줄 수도 있습니다.

```bash
python -m app.live_camera --camera 0 --device 0 --target-fps 10 --imgsz 640 --conf 0.25
```

카메라와 JSON 갱신만 짧게 확인하려면 화면 없이 몇 프레임만 실행할 수 있습니다.

```bash
python -m app.live_camera --no-display --max-frames 10
```

실행 중 화면에는 mask, bbox, label, 중심점, inference time, loop FPS가 표시됩니다. 종료하려면 표시 창에서 `q` 또는 `ESC`를 누릅니다.

최신 장면 상태는 기본적으로 아래 파일에 계속 갱신됩니다.

```text
runtime/latest_scene.json
```

또한 실행할 때마다 세션 단위 JSONL 로그가 생성됩니다.

```text
runtime/logs/live_camera_YYYYMMDD_HHMMSS.jsonl
```

`latest_scene.json`은 최신 상태만 덮어쓰고, JSONL 로그는 프레임별 scene state를 한 줄씩 누적합니다. 실험 문서화와 성능 분석은 JSONL 로그를 기준으로 합니다.

이 JSON들은 매 프레임의 full mask polygon을 제외한 compact 형태입니다. Language 모델로 넘기기 위한 기본 인터페이스로 사용합니다.

```json
{
  "frame_id": 128,
  "timestamp": "2026-05-05T14:00:00.000000+00:00",
  "camera_id": "front_rgb",
  "model": "yolo11s-seg.pt",
  "image_size": [1280, 720],
  "inference_ms": 24.3,
  "loop_fps": 9.8,
  "objects": [
    {
      "id": "obj_01",
      "label": "mouse",
      "confidence": 0.94,
      "bbox_xyxy": [331.23, 189.97, 971.52, 1228.74],
      "center_pixel": [655, 700],
      "area_pixels": 538823,
      "depth_m": null,
      "camera_xyz": null,
      "robot_xyz": null,
      "status": "detected"
    }
  ]
}
```

## 7. API 서버 실행

FastAPI 서버는 단발 이미지 테스트나 외부 모듈 연동용입니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

처음 실행할 때 Ultralytics가 `yolo11s-seg.pt` pretrained weight를 자동으로 내려받습니다.

## 8. API 테스트

full mask polygon을 포함한 JSON 응답:

```bash
curl -X POST "http://localhost:8000/predict" `
  -F "file=@path\to\image.jpg"
```

Language 모델 전달용 compact scene JSON 응답:

```bash
curl -X POST "http://localhost:8000/predict/scene" `
  -F "file=@path\to\image.jpg"
```

마스크, bbox, label이 그려진 PNG 응답:

```bash
curl -X POST "http://localhost:8000/predict/annotated" `
  -F "file=@path\to\image.jpg" `
  -o annotated.png
```

카메라 런타임이 갱신 중인 최신 scene JSON 조회:

```bash
curl "http://localhost:8000/scene/latest"
```

응답 예시는 다음과 같습니다.

```json
{
  "model": "yolo11s-seg.pt",
  "image_size": [640, 480],
  "objects": [
    {
      "id": "obj_01",
      "label": "cup",
      "confidence": 0.9321,
      "bbox_xyxy": [120.4, 80.2, 220.8, 260.7],
      "mask_polygon": [[125, 84], [218, 91], [212, 254], [130, 250]],
      "center_pixel": [169, 170],
      "area_pixels": 13452,
      "depth_m": null,
      "camera_xyz": null,
      "robot_xyz": null,
      "status": "detected"
    }
  ]
}
```

## 9. 참고

RTX 5060과 RTX PRO 6000 Blackwell은 NVIDIA Blackwell 계열입니다. CUDA 12.1 같은 오래된 PyTorch wheel 대신 CUDA 13 계열 PyTorch를 우선 사용합니다.

CUDA Toolkit이 시스템에 설치되어 있어도 PyTorch pip wheel은 자체 CUDA runtime을 포함합니다. 핵심은 `nvidia-smi`에서 보이는 driver가 설치한 PyTorch CUDA build를 지원하는지입니다.

공식 참고:

- NVIDIA CUDA GPU compute capability: https://developer.nvidia.com/cuda/gpus
- PyTorch install docs: https://docs.pytorch.org/get-started/locally/
- PyTorch CUDA 13 wheel index: https://download.pytorch.org/whl/cu130/

## 10. 다음 단계

1. Depth 카메라 입력을 `/predict`에 추가합니다.
2. `center_pixel + depth`를 camera XYZ로 변환합니다.
3. calibration matrix를 적용해 robot XYZ를 채웁니다.
4. 커스텀 tabletop 데이터셋으로 `red_block`, `left_box` 같은 클래스에 fine-tuning합니다.
