"""Pure rendering and export helpers for satellite imagery analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


MAX_OUTPUT_SIDE = 2048

LAND_COVER_PALETTE: dict[str, tuple[int, int, int]] = {
    "background": (45, 45, 45),
    "bareland": (210, 180, 140),
    "bare land": (210, 180, 140),
    "grass": (142, 202, 108),
    "pavement": (166, 166, 166),
    "road": (92, 92, 92),
    "tree": (34, 139, 34),
    "water": (45, 125, 210),
    "cropland": (236, 215, 91),
    "building": (218, 73, 73),
}


def normalize_label(label: str) -> str:
    return label.lower().replace("_", " ").strip()


def fallback_color(class_id: int) -> tuple[int, int, int]:
    return (
        int((67 * class_id + 41) % 190 + 35),
        int((97 * class_id + 73) % 190 + 35),
        int((43 * class_id + 109) % 190 + 35),
    )


def class_color(class_id: int, label: str) -> tuple[int, int, int]:
    return LAND_COVER_PALETTE.get(normalize_label(label), fallback_color(class_id))


def resize_for_inference(
    image: Image.Image,
    max_side: int = MAX_OUTPUT_SIDE,
) -> Image.Image:
    """Normalize orientation/RGB and bound memory while preserving aspect ratio."""
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    width, height = prepared.size
    longest = max(width, height)
    if longest <= max_side:
        return prepared
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return prepared.resize(size, resampling.LANCZOS)


def render_segmentation(
    image: Image.Image,
    class_map: np.ndarray,
    id2label: dict[int, str],
    opacity: float,
) -> tuple[Image.Image, Image.Image]:
    """Return a land-cover overlay and a categorical color mask."""
    height, width = class_map.shape
    color_array = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in np.unique(class_map):
        label = id2label.get(int(class_id), f"class_{int(class_id)}")
        color_array[class_map == class_id] = class_color(int(class_id), label)

    base = np.asarray(image.resize((width, height)), dtype=np.float32)
    overlay = (
        base * (1.0 - opacity) + color_array.astype(np.float32) * opacity
    ).astype(np.uint8)
    boundaries = np.zeros((height, width), dtype=bool)
    boundaries[1:, :] |= class_map[1:, :] != class_map[:-1, :]
    boundaries[:, 1:] |= class_map[:, 1:] != class_map[:, :-1]
    overlay[boundaries] = (255, 255, 255)
    return Image.fromarray(overlay), Image.fromarray(color_array)


def build_class_table(
    class_map: np.ndarray,
    id2label: dict[int, str],
    min_share_percent: float = 0.0,
) -> list[list[object]]:
    class_ids, counts = np.unique(class_map, return_counts=True)
    total_pixels = int(class_map.size)
    rows: list[list[object]] = []
    for class_id, count in zip(class_ids, counts):
        share = 100.0 * int(count) / total_pixels
        if share < min_share_percent:
            continue
        label = id2label.get(int(class_id), f"class_{int(class_id)}")
        color = class_color(int(class_id), label)
        rows.append(
            [
                int(class_id),
                label,
                int(count),
                round(share, 2),
                "#{:02X}{:02X}{:02X}".format(*color),
            ]
        )
    rows.sort(key=lambda row: float(row[3]), reverse=True)
    return rows


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_detections(
    image: Image.Image,
    detections: Iterable[dict[str, object]],
) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    short_side = min(rendered.size)
    line_width = max(2, round(short_side / 320))
    font = _load_font(max(12, min(24, round(short_side / 55))))
    for detection in sorted(
        detections,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    ):
        class_id = int(detection["class_id"])
        color = fallback_color(class_id)
        box = tuple(float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{detection['class_name']} {float(detection['confidence']):.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0] + 8
        text_height = text_box[3] - text_box[1] + 8
        left = max(0.0, min(box[0], rendered.width - text_width))
        top = max(0.0, box[1] - text_height)
        background = (left, top, left + text_width, top + text_height)
        draw.rectangle(background, fill=color)
        draw.text((left + 4, top + 4), label, fill="white", font=font)
    return rendered


def build_detection_summary(
    detections: Iterable[dict[str, object]],
) -> list[list[object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for detection in detections:
        grouped[str(detection["class_name"])].append(float(detection["confidence"]))
    rows = [
        [name, len(scores), round(sum(scores) / len(scores), 3), round(max(scores), 3)]
        for name, scores in grouped.items()
    ]
    rows.sort(key=lambda row: (-int(row[1]), str(row[0])))
    return rows


def build_detection_table(
    detections: Iterable[dict[str, object]],
    image_size: tuple[int, int],
) -> list[list[object]]:
    width, height = image_size
    rows: list[list[object]] = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        rows.append(
            [
                index,
                str(detection["class_name"]),
                round(float(detection["confidence"]), 3),
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1),
                round((x2 - x1) * (y2 - y1), 1),
                round(((x1 + x2) / 2) / width, 5),
                round(((y1 + y2) / 2) / height, 5),
            ]
        )
    return rows


def write_class_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "pixels", "share_percent", "color"])
        writer.writerows(rows)


def write_detection_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "object_id",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "area_pixels",
                "center_x_normalized",
                "center_y_normalized",
            ]
        )
        writer.writerows(rows)


def write_pixel_geojson(
    path: Path,
    detections: Iterable[dict[str, object]],
    image_size: tuple[int, int],
) -> None:
    """Write boxes as polygons in image-pixel coordinates, not geographic CRS."""
    width, height = image_size
    features = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        features.append(
            {
                "type": "Feature",
                "id": index,
                "properties": {
                    "class": str(detection["class_name"]),
                    "class_id": int(detection["class_id"]),
                    "confidence": round(float(detection["confidence"]), 6),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
                },
            }
        )
    collection = {
        "type": "FeatureCollection",
        "name": "satellite_detections_pixel_coordinates",
        "properties": {
            "coordinate_system": "image_pixels",
            "origin": "top_left",
            "image_width": width,
            "image_height": height,
        },
        "features": features,
    }
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
