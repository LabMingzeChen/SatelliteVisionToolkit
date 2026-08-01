#!/usr/bin/env python3
"""Render reproducible README graphics from committed case-result data."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "assets" / "results"
DATA_PATH = RESULTS_DIR / "urban_case_results.json"
CASE_GALLERY = [
    ("Dense residential", ROOT / "assets/cases/dense_residential.jpg"),
    ("Urban intersection", ROOT / "assets/cases/urban_intersection.jpg"),
    ("Marina / harbor", ROOT / "assets/cases/harbor_marina.jpg"),
    ("Parking lot", ROOT / "assets/cases/parking_lot.jpg"),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, radius: int = 20) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def render_gallery() -> None:
    canvas = Image.new("RGB", (1280, 360), "#F4F7F9")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 22), "Urban sample scenes", fill="#123746", font=font(30, True))
    draw.text((34, 60), "UC Merced / USGS · 256×256 RGB · approximately 0.3 m", fill="#607680", font=font(17))
    for index, (label, path) in enumerate(CASE_GALLERY):
        left = 34 + index * 310
        rounded(draw, (left, 96, left + 286, 338), "#FFFFFF", 16)
        image = Image.open(path).convert("RGB").resize((210, 190), Image.Resampling.LANCZOS)
        canvas.paste(image, (left + 38, 108))
        draw.text((left + 18, 306), label, fill="#173946", font=font(19, True))
    canvas.save(RESULTS_DIR / "urban_case_gallery.png", optimize=True)


def render_result(case: dict[str, object], model: str) -> None:
    canvas = Image.new("RGB", (1280, 540), "#F4F7F9")
    draw = ImageDraw.Draw(canvas)
    rounded(draw, (24, 24, 1256, 516), "#FFFFFF", 22)
    source = Image.open(ROOT / str(case["image"])).convert("RGB").resize((430, 430), Image.Resampling.LANCZOS)
    canvas.paste(source, (54, 55))

    predictions = list(case["predictions"])
    top_label, top_score = predictions[0]
    draw.text((530, 54), str(case["title"]), fill="#143746", font=font(31, True))
    draw.text((530, 94), "ACTUAL SCENE-LEVEL LULC OUTPUT", fill="#13806B", font=font(15, True))
    rounded(draw, (530, 126, 1218, 188), "#E9F7F3", 14)
    draw.text((550, 140), f"Top class  {top_label}", fill="#104B4B", font=font(23, True))
    draw.text((1010, 142), f"{float(top_score):.2f}%", fill="#0A665B", font=font(23, True))

    for index, (label, score) in enumerate(predictions):
        y = 218 + index * 47
        draw.text((530, y), str(label), fill="#304C56", font=font(17))
        rounded(draw, (720, y + 2, 1120, y + 24), "#E6EDF0", 11)
        width = max(4, round(400 * float(score) / 100))
        rounded(draw, (720, y + 2, 720 + width, y + 24), "#1B9A83", 11)
        draw.text((1140, y), f"{float(score):.2f}%", fill="#304C56", font=font(17, True))

    ambiguity = "moderate" if float(case["entropy"]) >= 0.35 else "low"
    draw.text(
        (530, 467),
        f"Normalized entropy {float(case['entropy']):.2f} ({ambiguity} ambiguity) · {float(case['elapsed_seconds']):.1f}s GPU inference",
        fill="#536B75",
        font=font(15),
    )
    draw.text((530, 490), f"Model: {model} · whole-scene hypothesis, not parcel ground truth", fill="#7A3E28", font=font(14))
    canvas.save(RESULTS_DIR / f"{case['key']}_result.png", optimize=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    render_gallery()
    for case in payload["cases"]:
        render_result(case, payload["model"])


if __name__ == "__main__":
    main()
