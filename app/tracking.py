from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from app.schemas import DetectedObject, PredictionResponse


@dataclass
class _TrackState:
    track_id: str
    label: str
    center_pixel: list[int]
    missed_frames: int = 0


class CentroidTracker:
    def __init__(self, *, max_distance_px: float = 120.0, max_missed_frames: int = 15) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self._next_track_number = 1
        self._tracks: dict[str, _TrackState] = {}

    def update(self, prediction: PredictionResponse) -> PredictionResponse:
        unmatched_track_ids = set(self._tracks)

        for detected in prediction.objects:
            track = self._nearest_track(detected, unmatched_track_ids)
            if track is None:
                track = self._create_track(detected)
            else:
                unmatched_track_ids.remove(track.track_id)

            track.center_pixel = detected.center_pixel
            track.missed_frames = 0
            detected.track_id = track.track_id
            detected.status = "tracked"

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        return prediction

    def _nearest_track(self, detected: DetectedObject, candidate_track_ids: set[str]) -> _TrackState | None:
        best_track: _TrackState | None = None
        best_distance = self.max_distance_px

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            if track.label != detected.label:
                continue

            distance = hypot(
                detected.center_pixel[0] - track.center_pixel[0],
                detected.center_pixel[1] - track.center_pixel[1],
            )
            if distance < best_distance:
                best_distance = distance
                best_track = track

        return best_track

    def _create_track(self, detected: DetectedObject) -> _TrackState:
        track_id = f"track_{self._next_track_number:03d}"
        self._next_track_number += 1
        track = _TrackState(
            track_id=track_id,
            label=detected.label,
            center_pixel=detected.center_pixel,
        )
        self._tracks[track_id] = track
        return track
