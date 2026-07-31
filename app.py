"""Hugging Face Space for satellite object detection and land-cover segmentation."""

from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def decorator(function):
                return function
            return decorator
    spaces = _SpacesFallback()

from satellite_utils import (
    build_class_table,
    build_detection_summary,
    build_detection_table,
    render_detections,
    render_segmentation,
    resize_for_inference,
    write_class_csv,
    write_detection_csv,
    write_pixel_geojson,
)


SEGMENTATION_MODEL_ID = "mfaytin/mask2former-satellite"
DETECTION_MODEL_ID = "bluelabel/satellite-equipment-detection-yolov8n-vhr10"
DETECTION_FILENAME = "best.pt"
OUTPUT_ROOT = Path("/tmp/satellite-vision-toolkit")
OPEN_EARTH_MAP_LABELS = {
    0: "background",
    1: "bareland",
    2: "grass",
    3: "pavement",
    4: "road",
    5: "tree",
    6: "water",
    7: "cropland",
    8: "building",
}


@lru_cache(maxsize=1)
def load_segmenter():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL_ID)
        .to(device)
        .eval()
    )
    return processor, model, OPEN_EARTH_MAP_LABELS, device


@lru_cache(maxsize=1)
def load_detector():
    from ultralytics import YOLO

    weights = hf_hub_download(repo_id=DETECTION_MODEL_ID, filename=DETECTION_FILENAME)
    return YOLO(weights)


def _new_output_dir() -> Path:
    output_dir = OUTPUT_ROOT / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@spaces.GPU(duration=120)
def segment_satellite_image(
    image: Image.Image | None,
    opacity: float,
    min_share_percent: float,
):
    if image is None:
        raise gr.Error("Please upload a satellite or aerial image first.")
    started_at = time.perf_counter()
    prepared = resize_for_inference(image)
    try:
        processor, model, id2label, device = load_segmenter()
        inputs = processor(images=prepared, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        class_map = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(prepared.height, prepared.width)],
        )[0].cpu().numpy().astype(np.uint8)
    except Exception as exc:
        raise gr.Error(f"Land-cover segmentation failed: {type(exc).__name__}: {exc}") from exc

    overlay, color_mask = render_segmentation(
        prepared, class_map, id2label, float(opacity)
    )
    rows = build_class_table(class_map, id2label, float(min_share_percent))
    output_dir = _new_output_dir()
    overlay_path = output_dir / "land_cover_overlay.png"
    mask_path = output_dir / "land_cover_color_mask.png"
    ids_path = output_dir / "land_cover_class_ids.png"
    csv_path = output_dir / "land_cover_summary.csv"
    overlay.save(overlay_path)
    color_mask.save(mask_path)
    Image.fromarray(class_map).save(ids_path)
    write_class_csv(csv_path, rows)
    elapsed = time.perf_counter() - started_at
    status = (
        f"Done · {prepared.width}×{prepared.height} · "
        f"{len(np.unique(class_map))} land-cover classes · {elapsed:.1f}s · device={device.type}"
    )
    return overlay, color_mask, rows, [str(overlay_path), str(mask_path), str(ids_path), str(csv_path)], status


@spaces.GPU(duration=120)
def detect_satellite_objects(
    image: Image.Image | None,
    confidence_threshold: float,
    iou_threshold: float,
):
    if image is None:
        raise gr.Error("Please upload a satellite or aerial image first.")
    started_at = time.perf_counter()
    prepared = resize_for_inference(image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        detector = load_detector()
        prediction = detector.predict(
            source=np.asarray(prepared),
            conf=float(confidence_threshold),
            iou=float(iou_threshold),
            imgsz=1024,
            device=device,
            max_det=500,
            verbose=False,
        )[0]
        detections: list[dict[str, object]] = []
        if prediction.boxes is not None:
            for coordinates, confidence, class_id_value in zip(
                prediction.boxes.xyxy.detach().cpu().tolist(),
                prediction.boxes.conf.detach().cpu().tolist(),
                prediction.boxes.cls.detach().cpu().tolist(),
            ):
                class_id = int(class_id_value)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": str(prediction.names[class_id]),
                        "confidence": float(confidence),
                        "x1": float(coordinates[0]),
                        "y1": float(coordinates[1]),
                        "x2": float(coordinates[2]),
                        "y2": float(coordinates[3]),
                    }
                )
    except Exception as exc:
        raise gr.Error(f"Satellite object detection failed: {type(exc).__name__}: {exc}") from exc

    overlay = render_detections(prepared, detections)
    summary_rows = build_detection_summary(detections)
    detail_rows = build_detection_table(detections, prepared.size)
    output_dir = _new_output_dir()
    overlay_path = output_dir / "satellite_detection_overlay.png"
    csv_path = output_dir / "satellite_detections.csv"
    geojson_path = output_dir / "satellite_detections_pixel_coordinates.geojson"
    overlay.save(overlay_path)
    write_detection_csv(csv_path, detail_rows)
    write_pixel_geojson(geojson_path, detections, prepared.size)
    elapsed = time.perf_counter() - started_at
    status = (
        f"Done · {prepared.width}×{prepared.height} · {len(detections)} objects · "
        f"{len(summary_rows)} classes · {elapsed:.1f}s · device={device}"
    )
    return overlay, summary_rows, detail_rows, [str(overlay_path), str(csv_path), str(geojson_path)], status


CSS = """
.gradio-container {max-width: 1280px !important;}
.hero {text-align: center; margin: 0 auto 1rem;}
.hero h1 {font-size: 2.15rem; margin-bottom: .3rem;}
.note {color: #64748b;}
"""

with gr.Blocks(title="Satellite Vision Toolkit", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>🛰️ Satellite Vision Toolkit</h1>
      <p>Detect remote-sensing objects and map land cover from one satellite or aerial image.</p>
      <p><a href="https://github.com/LabMingzeChen/SatelliteVisionToolkit">GitHub</a> ·
      <a href="https://huggingface.co/mfaytin/mask2former-satellite">Segmentation model</a> ·
      <a href="https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10">Detection model</a></p>
    </div>
    """)
    with gr.Row():
        image_input = gr.Image(type="pil", label="Satellite / aerial image", height=420)
        with gr.Column():
            gr.Markdown("""
### Supported analysis

- **Object detection:** airplane, ship, storage tank, sports fields/courts, harbor, bridge, and vehicle.
- **Land-cover segmentation:** background, bare land, grass, pavement, road, tree, water, cropland, and building.

RGB PNG/JPEG/WebP/TIFF images work best. Results are image-space estimates, not surveyed GIS data.
            """)

    with gr.Tabs():
        with gr.Tab("Land-cover segmentation"):
            with gr.Row():
                opacity = gr.Slider(0.1, 0.9, value=0.55, step=0.05, label="Overlay opacity")
                min_share = gr.Slider(0.0, 5.0, value=0.1, step=0.1, label="Minimum table share (%)")
            segment_button = gr.Button("Segment land cover", variant="primary")
            segment_status = gr.Markdown()
            with gr.Row():
                segment_overlay = gr.Image(label="Land-cover overlay")
                segment_mask = gr.Image(label="Categorical mask")
            segment_table = gr.Dataframe(
                headers=["Class ID", "Class", "Pixels", "Share (%)", "Color"],
                datatype=["number", "str", "number", "number", "str"],
                interactive=False,
                label="Land-cover area summary",
            )
            segment_files = gr.File(label="Download segmentation outputs", file_count="multiple")

        with gr.Tab("Object detection"):
            with gr.Row():
                confidence = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="NMS IoU threshold")
            detect_button = gr.Button("Detect satellite objects", variant="primary")
            detect_status = gr.Markdown()
            detect_overlay = gr.Image(label="Detection overlay")
            detection_summary = gr.Dataframe(
                headers=["Class", "Count", "Average confidence", "Maximum confidence"],
                datatype=["str", "number", "number", "number"],
                interactive=False,
                label="Detection summary",
            )
            detection_details = gr.Dataframe(
                headers=["ID", "Class", "Confidence", "x1", "y1", "x2", "y2", "Area (px²)", "Center x", "Center y"],
                interactive=False,
                label="Per-object results",
            )
            detection_files = gr.File(label="Download detection outputs", file_count="multiple")

    gr.Markdown("""
> **Responsible use:** Models can miss small objects and may generalize poorly across sensors, regions, seasons, cloud cover, and ground resolution. Pixel-coordinate GeoJSON is not georeferenced. Do not use outputs for navigation, surveillance, legal boundaries, emergency response, or other safety-critical decisions without qualified review.
    """)

    segment_button.click(
        segment_satellite_image,
        inputs=[image_input, opacity, min_share],
        outputs=[segment_overlay, segment_mask, segment_table, segment_files, segment_status],
        api_name="segment",
    )
    detect_button.click(
        detect_satellite_objects,
        inputs=[image_input, confidence, iou],
        outputs=[detect_overlay, detection_summary, detection_details, detection_files, detect_status],
        api_name="detect",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
