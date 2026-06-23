"""Foreground cutout: keep detected-object pixels, drop the background.

The perception loop already produces, per frame, an RGB image plus a
``PredictionResponse`` whose objects carry instance-segmentation polygons. This
module rasterizes the union of those polygons into a single-channel alpha mask
and applies it, yielding a "누끼" — the objects on a transparent background — so
a downstream consumer can reason about the objects without the background.

Design notes
------------
- Polygon rasterization is cheap (a handful of polygons per frame) and is meant
  to run on the perception thread. The expensive WEBP/PNG encoding is deferred
  to the cutout streamer's own thread so it never charges the YOLO loop budget.
- If an object has no usable polygon (e.g. a detection-only model like
  YOLO-World), its bounding box is filled instead — the cutout degrades to a
  rectangular crop rather than disappearing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.perception.schemas import PredictionResponse


def build_foreground_alpha(
    size: tuple[int, int], prediction: PredictionResponse
) -> Image.Image:
    """Single-channel (mode "L") mask: 255 on objects, 0 on background."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for detected in prediction.objects:
        if len(detected.mask_polygon) >= 3:
            draw.polygon([tuple(point) for point in detected.mask_polygon], fill=255)
        elif detected.bbox_xyxy:
            x1, y1, x2, y2 = (int(round(value)) for value in detected.bbox_xyxy)
            draw.rectangle((x1, y1, x2, y2), fill=255)
    return mask


def build_foreground_cutout_rgba(
    frame_rgb: np.ndarray, prediction: PredictionResponse
) -> np.ndarray:
    """Return an ``H x W x 4`` RGBA array: object pixels opaque, background transparent.

    ``frame_rgb`` is an ``H x W x 3`` uint8 RGB array (as produced by the capture
    loop). A fresh array is allocated, so the caller may keep using ``frame_rgb``.
    """
    height, width = frame_rgb.shape[:2]
    alpha = np.asarray(build_foreground_alpha((width, height), prediction), dtype=np.uint8)
    return np.dstack((frame_rgb, alpha))


def build_foreground_cutout(
    image: Image.Image, prediction: PredictionResponse, *, background: tuple[int, int, int] | None = None
) -> Image.Image:
    """PIL convenience wrapper used by the HTTP endpoint.

    ``background=None`` returns an RGBA image with a transparent background.
    Passing an RGB color flattens the cutout onto it and returns RGB (useful for
    consumers that ignore the alpha channel).
    """
    base = image.convert("RGB")
    alpha = build_foreground_alpha(base.size, prediction)
    if background is None:
        rgba = base.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba
    flat = Image.new("RGB", base.size, background)
    flat.paste(base, (0, 0), alpha)
    return flat
