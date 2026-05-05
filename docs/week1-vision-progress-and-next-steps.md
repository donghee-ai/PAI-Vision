# Vision Progress and Next Steps

## 오늘 완료한 것

PAI-Vision 레포에서 YOLO11-Seg 기반 Vision MVP를 구성했다.

구현된 흐름:

```text
Camera frame
→ YOLO11-Seg inference
→ segmentation overlay display
→ compact scene JSON
→ latest scene snapshot
→ session JSONL log
```

## 구현 상태

### 1. GPU 환경

- 로컬 RTX 5060에서 CUDA 기반 PyTorch 실행을 확인했다.
- `torch==2.10.0+cu130`, CUDA runtime `13.0` 환경에서 동작했다.
- `requirements.txt`에는 `torch`를 넣지 않고, 환경별로 PyTorch를 따로 설치하는 방향으로 정리했다.

### 2. YOLO11-Seg 추론

- pretrained `yolo11s-seg.pt`를 사용했다.
- 이미지 단발 추론, API 추론, 카메라 실시간 추론을 모두 확인했다.
- COCO pretrained 기준으로 객체 label, confidence, bbox, mask, center pixel이 정상 생성된다.

### 3. 실시간 카메라 루프

- `python -m app.live_camera`로 카메라 실시간 추론을 실행할 수 있다.
- 화면에는 segmentation mask, bbox, label, center point, inference time, loop FPS가 표시된다.
- 목표 FPS는 `.env`의 `CAMERA_TARGET_FPS=10`으로 설정했다.

### 4. Scene JSON

Language/Action 파트가 읽기 쉬운 compact scene JSON을 만들었다.

최신 상태 파일:

```text
runtime/latest_scene.json
```

기본 형태:

```json
{
  "frame_id": 300,
  "timestamp": "2026-05-05T14:49:32.112332+00:00",
  "camera_id": "front_rgb",
  "model": "yolo11s-seg.pt",
  "image_size": [1280, 720],
  "inference_ms": 19.08,
  "loop_fps": 9.98,
  "objects": []
}
```

### 5. 세션 로그

실험마다 프레임별 scene JSON을 JSONL로 누적 저장한다.

```text
runtime/logs/live_camera_YYYYMMDD_HHMMSS.jsonl
```

이 로그는 이후 실험 문서화, FPS 분석, 객체 분포 분석, Language 모델 입력 재현에 사용할 수 있다.

### 6. 실험 문서

아래 문서를 작성했다.

```text
docs/experiments/2026-05-05-live-camera-smoke-test.md
docs/experiments/2026-05-05-live-camera-session-02.md
```

2차 실험 요약:

| 항목 | 결과 |
| --- | ---: |
| frames | 300 |
| avg inference | 23.14 ms |
| p50 inference | 20.26 ms |
| p95 inference | 30.28 ms |
| avg loop FPS | 9.75 |
| p50 loop FPS | 9.99 |
| frames with objects | 294 / 300 |

## 현재 한계

### 1. Depth 없음

현재 YOLO는 RGB 이미지에서 2D 정보만 제공한다.

가능한 정보:

```text
label
confidence
bbox
mask
center_pixel
```

아직 없는 정보:

```text
depth_m
camera_xyz
robot_xyz
grasp_hint
```

이 값들은 YOLO가 기본 제공하지 않는다. RGB-D 카메라, stereo depth, table calibration, 또는 별도 depth model이 필요하다.

### 2. Robot 좌표계 없음

로봇이 실제로 움직이려면 `center_pixel`이 아니라 로봇 기준 3D 좌표가 필요하다.

필요한 변환:

```text
pixel coordinate
→ depth
→ camera coordinate
→ robot coordinate
```

이를 위해 camera intrinsics와 camera-to-robot extrinsics가 필요하다.

### 3. Pretrained COCO label

현재 모델은 COCO pretrained라서 tabletop 데모 객체에 최적화되어 있지 않다.

예상 프로젝트 객체:

```text
red_block
blue_block
yellow_cup
left_box
right_box
```

이런 객체는 커스텀 데이터셋으로 fine-tuning해야 안정적으로 인식된다.

### 4. Object ID 안정성

현재 `obj_01`, `obj_02`는 프레임마다 새로 붙는 단일 프레임 ID다.

Language/Action 파트와 안정적으로 연결하려면 프레임 간 tracking ID가 필요하다.

## 앞으로 해야 할 것

### 1. 팀 회의에서 정할 것

- Vision 결과를 Language/Action으로 전달하는 방식
- WebSocket을 먼저 쓸지, ROS2 bridge를 고려할지
- 실제 카메라와 로봇 구성
- 초기 데모 객체 클래스
- Depth 카메라 사용 여부
- MuJoCo 시뮬레이션과 실제 로봇 연결 방식

### 2. 통신 구조

현재 HTTP API는 테스트와 디버그에는 충분하다.

실시간 VLA 연결을 위해서는 다음 중 하나를 정해야 한다.

| 방식 | 용도 |
| --- | --- |
| WebSocket | scene JSON을 실시간 stream으로 전달 |
| ROS2 topic | 실제 로봇 시스템과 통합 |
| file polling | 가장 단순한 MVP 방식 |
| queue/pub-sub | 프로세스 간 안정적 전달 |

추천 순서:

```text
WebSocket scene stream
→ Language planner 연결
→ 이후 ROS2 bridge 검토
```

### 3. Depth / 3D 좌표

Depth 카메라가 들어오면 다음 작업을 한다.

```text
YOLO mask
→ mask 내부 depth median
→ center_pixel + depth_m
→ camera_xyz
→ robot_xyz
```

추가해야 할 필드:

```json
{
  "depth_m": 0.64,
  "camera_xyz": [0.12, -0.05, 0.64],
  "robot_xyz": [0.41, 0.18, 0.09]
}
```

### 4. 커스텀 데이터셋

초기 데모 객체를 제한하고 segmentation dataset을 만든다.

예상 클래스:

```text
red_block
blue_block
yellow_cup
left_box
right_box
```

수집 조건:

```text
밝은 조명 / 어두운 조명
정면 / 측면 / 위쪽 시점
단일 물체 / 여러 물체
겹침 있음 / 겹침 없음
배경 단순 / 배경 복잡
```

### 5. 모델 비교

실시간성과 정확도를 비교한다.

| 모델 | 목적 |
| --- | --- |
| YOLO11n-seg | 더 빠른 실시간 MVP |
| YOLO11s-seg | 현재 기준 모델 |
| YOLO11m-seg | 정확도 비교 |

### 6. 로그 분석 자동화

현재는 JSONL 로그를 사람이 요청하면 문서로 정리한다.

다음에는 자동 분석 스크립트를 추가할 수 있다.

분석 항목:

```text
avg inference_ms
p50/p95 inference_ms
avg loop_fps
object count per frame
label distribution
confidence distribution
frames with/without objects
```

## 현재 결론

Vision 파트 MVP는 성공했다.

현재 시스템은 다음을 안정적으로 수행한다.

```text
카메라 실시간 입력
YOLO11-Seg GPU 추론
화면 overlay
VLA용 scene JSON 생성
세션 로그 저장
실험 결과 문서화
```

다음 핵심 단계는 연결 방식 결정과 hardware 기반 3D 좌표 확장이다.

```text
단기: WebSocket scene stream 또는 Language planner 연결
중기: custom YOLO11-Seg fine-tuning
장기: Depth + camera_xyz + robot_xyz + ROS2/robot integration
```
