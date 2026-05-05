# PAI-Vision

Physical AI 기반 VLA 파이프라인의 Vision 모듈 실험 레포입니다.

현재 버전은 pretrained `YOLO11s-seg` 모델을 FastAPI로 서빙해서 이미지 입력을 segmentation JSON으로 변환합니다.

## 1. 환경 전략

이 프로젝트에서는 `torch`, `torchvision`, `torchaudio`를 `requirements.txt`에 넣지 않습니다.

이유는 실행 환경마다 CUDA/PyTorch 조합이 다르기 때문입니다.

| 환경 | 권장 방식 |
| --- | --- |
| 로컬 RTX 5060 | `.venv` + CUDA 13.0 PyTorch wheel |
| RTX PRO 6000 Blackwell 서버 | 기존 CUDA 13 PyTorch Docker 이미지 |
| 다른 GPU 서버 | Docker base image로 CUDA/PyTorch 고정 |
| Docker를 못 쓰는 서버 | conda 또는 `.venv` 대안 사용 |

`requirements.txt`는 FastAPI, Ultralytics, Pillow 같은 앱 공통 의존성만 관리합니다.

## 2. 로컬 RTX 5060 실행 준비

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

```env
YOLO_MODEL=yolo11s-seg.pt
YOLO_DEVICE=0
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_IOU=0.7
```

여러 GPU가 있으면 `YOLO_DEVICE=1`, `YOLO_DEVICE=2`처럼 바꿔서 특정 GPU를 지정할 수 있습니다.

## 3. CPU Torch가 깔렸을 때 복구

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

## 4. RTX PRO 6000 Blackwell 서버

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

## 5. 서버 실행

로컬 기본 실행:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

처음 실행할 때 Ultralytics가 `yolo11s-seg.pt` pretrained weight를 자동으로 내려받습니다.

## 6. 테스트

```bash
curl -X POST "http://localhost:8000/predict" `
  -F "file=@path\to\image.jpg"
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

## 7. 참고

RTX 5060과 RTX PRO 6000 Blackwell은 NVIDIA Blackwell 계열입니다. CUDA 12.1 같은 오래된 PyTorch wheel 대신 CUDA 13 계열 PyTorch를 우선 사용합니다.

CUDA Toolkit이 시스템에 설치되어 있어도 PyTorch pip wheel은 자체 CUDA runtime을 포함합니다. 핵심은 `nvidia-smi`에서 보이는 driver가 설치한 PyTorch CUDA build를 지원하는지입니다.

공식 참고:

- NVIDIA CUDA GPU compute capability: https://developer.nvidia.com/cuda/gpus
- PyTorch install docs: https://docs.pytorch.org/get-started/locally/
- PyTorch CUDA 13 wheel index: https://download.pytorch.org/whl/cu130/

## 8. 다음 단계

1. Depth 카메라 입력을 `/predict`에 추가합니다.
2. `center_pixel + depth`를 camera XYZ로 변환합니다.
3. calibration matrix를 적용해 robot XYZ를 채웁니다.
4. 커스텀 tabletop 데이터셋으로 `red_block`, `left_box` 같은 클래스에 fine-tuning합니다.
