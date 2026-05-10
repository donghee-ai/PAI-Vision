# Live Camera Session 02 - 2026-05-05

## Summary

세션 단위 JSONL 로그를 활성화한 뒤, 로컬 RTX 5060 환경에서 `YOLO11s-seg` 카메라 실시간 추론을 다시 실행했다.

이번 실험의 목적은 다음 두 가지였다.

1. 10 FPS 목표 실시간 루프가 안정적으로 유지되는지 확인한다.
2. 프레임별 compact scene JSON이 session log로 누적 저장되는지 확인한다.

결과적으로 300프레임 로그가 정상 저장되었고, 평균 loop FPS는 `9.75`, p50 loop FPS는 `9.99`로 10 FPS 목표에 거의 도달했다.

## Source Log

```text
runtime/logs/live_camera_20260505_234901.jsonl
```

## Environment

| Item | Value |
| --- | --- |
| OS | Windows |
| GPU | NVIDIA GeForce RTX 5060 |
| PyTorch | `2.10.0+cu130` |
| CUDA runtime | `13.0` |
| CUDA available | `True` |
| OpenCV | `4.13.0` |
| Ultralytics | `8.4.46` |
| Model | `yolo11s-seg.pt` |

## Runtime Settings

| Field | Value |
| --- | --- |
| `camera_id` | `front_rgb` |
| `image_size` | `[1280, 720]` |
| `target_fps` | `10` |
| `imgsz` | `640` |
| `conf` | `0.25` |
| `iou` | `0.7` |
| `scene_json_path` | `runtime/latest_scene.json` |
| `scene_log_dir` | `runtime/logs` |

## Session Metrics

| Metric | Value |
| --- | ---: |
| frames | 300 |
| first timestamp | `2026-05-05T14:49:02.071941+00:00` |
| last timestamp | `2026-05-05T14:49:32.112332+00:00` |
| frames with objects | 294 |
| frames without objects | 6 |

Inference time:

| Metric | ms |
| --- | ---: |
| min | 11.75 |
| avg | 23.14 |
| p50 | 20.26 |
| p95 | 30.28 |
| max | 620.89 |

Loop FPS:

| Metric | FPS |
| --- | ---: |
| min | 1.09 |
| avg | 9.75 |
| p50 | 9.99 |
| p95 | 10.33 |
| max | 11.53 |

Object count per frame:

| Metric | Count |
| --- | ---: |
| min | 0 |
| avg | 6.27 |
| p50 | 6 |
| p95 | 12 |
| max | 15 |

## Label Distribution

| Label | Count | Avg Confidence |
| --- | ---: | ---: |
| person | 689 | 0.6315 |
| potted plant | 673 | 0.4623 |
| bench | 270 | 0.3777 |
| bicycle | 110 | 0.7042 |
| fire hydrant | 34 | 0.5453 |
| chair | 18 | 0.4259 |
| stop sign | 17 | 0.3430 |
| sports ball | 17 | 0.3149 |
| handbag | 14 | 0.2804 |
| surfboard | 10 | 0.3562 |
| car | 8 | 0.3559 |
| umbrella | 8 | 0.3493 |
| clock | 6 | 0.3125 |
| teddy bear | 3 | 0.4957 |
| parking meter | 1 | 0.3339 |
| cup | 1 | 0.5964 |
| bus | 1 | 0.3677 |
| vase | 1 | 0.3437 |
| motorcycle | 1 | 0.4585 |

## Last Scene Snapshot

마지막 프레임은 객체가 없는 상태로 기록되었다.

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

## Interpretation

- 300프레임 동안 session JSONL 로그가 정상적으로 누적되었다.
- 평균 loop FPS `9.75`, p50 loop FPS `9.99`로 10 FPS 목표에 거의 맞았다.
- 평균 inference time `23.14 ms`, p95 inference time `30.28 ms`로 GPU 실시간 추론 여유가 확인되었다.
- max inference time `620.89 ms`는 모델 warmup, 카메라/OS scheduling, 또는 초기 프레임 처리 영향일 가능성이 있다.
- 마지막 프레임에는 객체가 없었지만, 전체 300프레임 중 294프레임에서 객체가 탐지되었다.
- COCO pretrained 모델이므로 일부 label은 실제 환경 물체와 다르게 매칭될 수 있다.

## VLA Pipeline Notes

이번 로그는 Language 모델에 넘기기 위한 scene interface의 기본 형태를 검증했다.

현재 scene JSON은 다음 목적에 적합하다.

- 최신 장면 상태 polling
- local adapter 또는 외부 hub의 scene stream
- Language planner 입력
- 추후 별도 ROS2 bridge의 `/vision/scene` topic 변환

아직 필요한 항목:

1. 프레임 간 object tracking ID 안정화
2. Depth 기반 `camera_xyz` 계산
3. robot calibration 기반 `robot_xyz` 계산
4. tabletop 커스텀 클래스 fine-tuning
5. local adapter stream은 개발용으로 유지하고, 장기 stream/bridge는 외부 hub 쪽에서 소유
