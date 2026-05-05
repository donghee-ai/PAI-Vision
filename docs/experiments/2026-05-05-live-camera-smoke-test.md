# Live Camera Smoke Test - 2026-05-05

## Summary

Pretrained `YOLO11s-seg`를 로컬 RTX 5060 환경에서 카메라 입력으로 실행했다.

목표는 웹서버 단발 추론이 아니라, 카메라 프레임을 계속 받아 segmentation 결과를 화면에 overlay하고, Language/Action 파트가 읽을 수 있는 최신 scene JSON을 갱신하는 것이었다.

결과적으로 카메라 루프, GPU 추론, 화면 overlay, `runtime/latest_scene.json` 갱신이 정상 동작했다.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows |
| GPU | NVIDIA GeForce RTX 5060 |
| Python | `.venv` |
| PyTorch | `2.10.0+cu130` |
| CUDA runtime | `13.0` |
| CUDA available | `True` |
| OpenCV | `4.13.0` |
| Ultralytics | `8.4.46` |
| Model | `yolo11s-seg.pt` |

## Runtime Settings

```env
YOLO_MODEL=yolo11s-seg.pt
YOLO_DEVICE=0
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_IOU=0.7
CAMERA_ID=front_rgb
CAMERA_INDEX=0
CAMERA_TARGET_FPS=10
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
SCENE_JSON_PATH=runtime/latest_scene.json
```

## Commands

Interactive display:

```powershell
python -m app.live_camera
```

Headless smoke test:

```powershell
python -m app.live_camera --no-display --max-frames 5 --target-fps 10
```

## Latest Scene Snapshot

`runtime/latest_scene.json` 기준 마지막 확인 프레임:

| Field | Value |
| --- | --- |
| `frame_id` | `5` |
| `timestamp` | `2026-05-05T14:32:04.300485+00:00` |
| `camera_id` | `front_rgb` |
| `image_size` | `[1280, 720]` |
| `inference_ms` | `33.49` |
| object count | `11` |

Detected object summary:

| id | label | confidence | center_pixel | area_pixels |
| --- | --- | ---: | --- | ---: |
| obj_01 | person | 0.6997 | `[196, 534]` | 19338 |
| obj_02 | person | 0.6844 | `[942, 309]` | 1136 |
| obj_03 | person | 0.4899 | `[1010, 322]` | 521 |
| obj_04 | bench | 0.4573 | `[507, 658]` | 25752 |
| obj_05 | person | 0.4518 | `[197, 550]` | 21459 |
| obj_06 | person | 0.3725 | `[645, 485]` | 4866 |
| obj_07 | bench | 0.3483 | `[66, 386]` | 13026 |
| obj_08 | bench | 0.3342 | `[30, 419]` | 2797 |
| obj_09 | bench | 0.3148 | `[976, 395]` | 28483 |
| obj_10 | bench | 0.2936 | `[853, 663]` | 25302 |
| obj_11 | surfboard | 0.2915 | `[259, 517]` | 2937 |

## Observations

- GPU 추론은 정상 동작했다.
- 실시간 화면 overlay도 정상 동작했다.
- 마지막 smoke test에서 single-frame inference time은 `33.49 ms`였다.
- GPU 자원 사용량은 체감상 크지 않았다.
- `runtime/latest_scene.json`은 Language 모델로 넘기기 위한 compact scene state로 적합하다.
- 현재 객체 label은 COCO pretrained 기준이므로 tabletop 커스텀 객체 인식에는 fine-tuning이 필요하다.

## Issue Found

Windows에서 `runtime/latest_scene.json.tmp`를 `runtime/latest_scene.json`으로 교체하는 과정에서 `PermissionError: WinError 5`가 발생했다.

원인은 VS Code, curl, 파일 미리보기 등 다른 프로세스가 `latest_scene.json`을 읽는 순간 Windows 파일 잠금이 걸렸을 가능성이 높다.

대응:

- temp 파일명을 매번 고유하게 생성하도록 수정했다.
- replace 실패 시 짧게 재시도하도록 수정했다.
- 재시도 후에도 실패하면 카메라 루프를 죽이지 않고 해당 프레임의 scene write만 skip하도록 수정했다.

## Next Steps

1. `runtime/latest_scene.json` 외에 session별 `.jsonl` 누적 로그를 저장한다.
2. `/scene/stream` WebSocket으로 scene JSON을 실시간 전달한다.
3. mask polygon은 실시간 Language 입력에서는 제외하고, 필요할 때만 별도 조회한다.
4. Depth 입력을 추가해서 `camera_xyz`, `robot_xyz` 필드를 채운다.
5. 이후 ROS2 bridge를 추가해 `/vision/scene` topic으로 publish할 수 있게 한다.
