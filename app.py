"""Professional Hugging Face Space for multi-level satellite image analysis."""

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
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Mask2FormerForUniversalSegmentation,
)

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
    build_analysis_summary,
    build_class_table,
    build_detection_summary,
    build_detection_table,
    build_lulc_table,
    normalized_entropy,
    render_detections,
    render_lulc_assessment,
    render_segmentation,
    resize_for_inference,
    write_class_csv,
    write_detection_csv,
    write_json,
    write_lulc_csv,
    write_pixel_geojson,
)


CLASSIFICATION_MODEL_ID = "mrm8488/convnext-tiny-finetuned-eurosat"
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


def _device() -> torch.device:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_classifier():
    device = _device()
    processor = AutoImageProcessor.from_pretrained(CLASSIFICATION_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(CLASSIFICATION_MODEL_ID).to(device).eval()
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    return processor, model, id2label, device


@lru_cache(maxsize=1)
def load_segmenter():
    device = _device()
    processor = AutoImageProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL_ID).to(device).eval()
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


def _require_image(image: Image.Image | None) -> Image.Image:
    if image is None:
        raise gr.Error("Please upload a satellite or aerial image first.")
    return resize_for_inference(image)


def _classify_impl(prepared: Image.Image, top_k: int, output_dir: Path) -> dict[str, object]:
    processor, model, id2label, device = load_classifier()
    inputs = {name: tensor.to(device) for name, tensor in processor(images=prepared, return_tensors="pt").items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    rows = build_lulc_table(probabilities, id2label, top_k)
    entropy = normalized_entropy(probabilities)
    csv_path = output_dir / "lulc_classification.csv"
    json_path = output_dir / "lulc_classification.json"
    write_lulc_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "task": "scene_level_lulc_classification",
            "model": CLASSIFICATION_MODEL_ID,
            "processed_image_size": {"width": prepared.width, "height": prepared.height},
            "normalized_entropy": round(entropy, 6),
            "predictions": [
                {"rank": row[0], "class": row[1], "probability_percent": row[2], "confidence_tier": row[3]}
                for row in rows
            ],
            "scope_note": "Whole-scene EuroSAT class; not a cadastral or planning land-use designation.",
        },
    )
    return {
        "rows": rows,
        "entropy": entropy,
        "assessment": render_lulc_assessment(rows, entropy),
        "files": [str(csv_path), str(json_path)],
        "device": device.type,
    }


def _segment_impl(
    prepared: Image.Image,
    opacity: float,
    min_share_percent: float,
    output_dir: Path,
) -> dict[str, object]:
    processor, model, id2label, device = load_segmenter()
    inputs = {name: tensor.to(device) for name, tensor in processor(images=prepared, return_tensors="pt").items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    class_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(prepared.height, prepared.width)],
    )[0].cpu().numpy().astype(np.uint8)
    overlay, color_mask = render_segmentation(prepared, class_map, id2label, float(opacity))
    rows = build_class_table(class_map, id2label, float(min_share_percent))
    overlay_path = output_dir / "land_cover_overlay.png"
    mask_path = output_dir / "land_cover_color_mask.png"
    ids_path = output_dir / "land_cover_class_ids.png"
    csv_path = output_dir / "land_cover_summary.csv"
    overlay.save(overlay_path)
    color_mask.save(mask_path)
    Image.fromarray(class_map).save(ids_path)
    write_class_csv(csv_path, rows)
    return {
        "overlay": overlay,
        "mask": color_mask,
        "rows": rows,
        "files": [str(overlay_path), str(mask_path), str(ids_path), str(csv_path)],
        "device": device.type,
    }


def _detect_impl(
    prepared: Image.Image,
    confidence_threshold: float,
    iou_threshold: float,
    output_dir: Path,
) -> dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    overlay = render_detections(prepared, detections)
    summary_rows = build_detection_summary(detections)
    detail_rows = build_detection_table(detections, prepared.size)
    overlay_path = output_dir / "satellite_detection_overlay.png"
    csv_path = output_dir / "satellite_detections.csv"
    geojson_path = output_dir / "satellite_detections_pixel_coordinates.geojson"
    overlay.save(overlay_path)
    write_detection_csv(csv_path, detail_rows)
    write_pixel_geojson(geojson_path, detections, prepared.size)
    return {
        "overlay": overlay,
        "summary": summary_rows,
        "details": detail_rows,
        "files": [str(overlay_path), str(csv_path), str(geojson_path)],
        "device": device,
    }


@spaces.GPU(duration=120)
def classify_lulc(image: Image.Image | None, top_k: int):
    started_at = time.perf_counter()
    prepared = _require_image(image)
    try:
        result = _classify_impl(prepared, int(top_k), _new_output_dir())
    except Exception as exc:
        raise gr.Error(f"LULC classification failed: {type(exc).__name__}: {exc}") from exc
    status = (
        f"Complete · {prepared.width}×{prepared.height} · top class {result['rows'][0][1]} "
        f"({result['rows'][0][2]:.1f}%) · {time.perf_counter() - started_at:.1f}s · device={result['device']}"
    )
    return result["assessment"], result["rows"], result["files"], status


@spaces.GPU(duration=120)
def segment_satellite_image(image: Image.Image | None, opacity: float, min_share_percent: float):
    started_at = time.perf_counter()
    prepared = _require_image(image)
    try:
        result = _segment_impl(prepared, opacity, min_share_percent, _new_output_dir())
    except Exception as exc:
        raise gr.Error(f"Land-cover segmentation failed: {type(exc).__name__}: {exc}") from exc
    status = (
        f"Complete · {prepared.width}×{prepared.height} · {len(result['rows'])} reported cover classes · "
        f"{time.perf_counter() - started_at:.1f}s · device={result['device']}"
    )
    return result["overlay"], result["mask"], result["rows"], result["files"], status


@spaces.GPU(duration=120)
def detect_satellite_objects(image: Image.Image | None, confidence_threshold: float, iou_threshold: float):
    started_at = time.perf_counter()
    prepared = _require_image(image)
    try:
        result = _detect_impl(prepared, confidence_threshold, iou_threshold, _new_output_dir())
    except Exception as exc:
        raise gr.Error(f"Satellite object detection failed: {type(exc).__name__}: {exc}") from exc
    status = (
        f"Complete · {prepared.width}×{prepared.height} · {len(result['details'])} objects · "
        f"{len(result['summary'])} classes · {time.perf_counter() - started_at:.1f}s · device={result['device']}"
    )
    return result["overlay"], result["summary"], result["details"], result["files"], status


@spaces.GPU(duration=180)
def analyze_satellite_image(
    image: Image.Image | None,
    top_k: int,
    opacity: float,
    min_share_percent: float,
    confidence_threshold: float,
    iou_threshold: float,
):
    started_at = time.perf_counter()
    prepared = _require_image(image)
    output_dir = _new_output_dir()
    try:
        classification = _classify_impl(prepared, int(top_k), output_dir)
        segmentation = _segment_impl(prepared, opacity, min_share_percent, output_dir)
        detection = _detect_impl(prepared, confidence_threshold, iou_threshold, output_dir)
    except Exception as exc:
        raise gr.Error(f"Complete analysis failed: {type(exc).__name__}: {exc}") from exc
    elapsed = time.perf_counter() - started_at
    summary = build_analysis_summary(
        classification["rows"],
        float(classification["entropy"]),
        segmentation["rows"],
        detection["summary"],
        elapsed,
    )
    report_path = output_dir / "analysis_report.json"
    write_json(
        report_path,
        {
            "processed_image_size": {"width": prepared.width, "height": prepared.height},
            "models": {
                "classification": CLASSIFICATION_MODEL_ID,
                "segmentation": SEGMENTATION_MODEL_ID,
                "detection": DETECTION_MODEL_ID,
            },
            "lulc_classification": classification["rows"],
            "lulc_normalized_entropy": round(float(classification["entropy"]), 6),
            "land_cover_pixel_shares": segmentation["rows"],
            "detection_summary": detection["summary"],
            "detection_details": detection["details"],
            "elapsed_seconds": round(elapsed, 3),
            "coordinate_note": "Detection GeoJSON is in top-left-origin image pixels and has no geographic CRS.",
        },
    )
    files = classification["files"] + segmentation["files"] + detection["files"] + [str(report_path)]
    status = f"Complete multi-model assessment · {prepared.width}×{prepared.height} · {elapsed:.1f}s"
    return (
        summary,
        classification["assessment"],
        classification["rows"],
        segmentation["overlay"],
        segmentation["mask"],
        segmentation["rows"],
        detection["overlay"],
        detection["summary"],
        detection["details"],
        files,
        status,
    )


CSS = """
.gradio-container {max-width: 1440px !important; background: #f6f8fb;}
.hero {padding: 2rem; border-radius: 22px; color: white; background: linear-gradient(125deg,#071c33,#0a4b5c 56%,#198f75); box-shadow: 0 18px 44px rgba(7,28,51,.18); margin-bottom: 1rem;}
.hero h1 {font-size: 2.35rem; margin: 0 0 .35rem; letter-spacing: -.03em;}
.hero p {max-width: 850px; margin: .35rem 0; color: #d8f3ee;}
.hero a {color: #fff; font-weight: 650;}
.pipeline {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0 20px;}
.pipeline div,.assessment-card,.metric-card {background:white;border:1px solid #dce6ed;border-radius:16px;padding:16px;box-shadow:0 6px 18px rgba(20,50,70,.06);}
.pipeline b {display:block;color:#0c5262;margin-bottom:5px}.pipeline span,.micro-note {color:#667985;font-size:.87rem;}
.summary-grid {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0;}
.metric-card span,.eyebrow {display:block;color:#66808c;font-size:.72rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase;}
.metric-card strong {display:block;font-size:1.45rem;margin:6px 0;color:#113544;}.metric-card small {color:#60747e;}
.assessment-card h2 {margin:.25rem 0;color:#123c49}.assessment-card p {color:#526b76;}
.prob-row {display:grid;grid-template-columns:155px 1fr 62px;gap:10px;align-items:center;margin:8px 0;font-size:.86rem;}
.prob-row b {text-align:right}.prob-track {height:9px;background:#e5edf1;border-radius:20px;overflow:hidden}.prob-track i {display:block;height:100%;background:linear-gradient(90deg,#169c7d,#36b7c5);border-radius:20px;}
.section-note {padding:12px 14px;border-left:4px solid #15947a;background:#eef9f6;border-radius:8px;color:#315c62;}
@media(max-width:850px){.pipeline,.summary-grid{grid-template-columns:1fr}.prob-row{grid-template-columns:115px 1fr 56px}}
"""


with gr.Blocks(title="Satellite Vision Toolkit Pro", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero">
      <div class="eyebrow" style="color:#8ee5d2">REMOTE SENSING DECISION SUPPORT</div>
      <h1>🛰️ Satellite Vision Toolkit Pro</h1>
      <p>A multi-level workbench for scene-level land-use/land-cover classification, pixel-level cover mapping, and overhead object detection.</p>
      <p><a href="https://github.com/LabMingzeChen/SatelliteVisionToolkit">GitHub</a> · <a href="https://huggingface.co/mrm8488/convnext-tiny-finetuned-eurosat">LULC model</a> · <a href="https://huggingface.co/mfaytin/mask2former-satellite">Segmentation model</a> · <a href="https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10">Detection model</a></p>
    </div>
    <div class="pipeline">
      <div><b>01 · Scene classification</b><span>EuroSAT probability profile across 10 LULC scene types.</span></div>
      <div><b>02 · Semantic segmentation</b><span>Per-pixel OpenEarthMap land-cover composition and masks.</span></div>
      <div><b>03 · Object detection</b><span>Bounding boxes and inventory-style summaries for 10 VHR object types.</span></div>
    </div>
    """)
    with gr.Row(equal_height=True):
        image_input = gr.Image(type="pil", label="Satellite / aerial RGB image", height=430)
        with gr.Column():
            gr.Markdown("### Analysis controls\nTune reproducible thresholds, then run the complete assessment or an individual method.")
            top_k = gr.Slider(3, 10, value=5, step=1, label="LULC alternatives (top-k)")
            opacity = gr.Slider(0.1, 0.9, value=0.55, step=0.05, label="Segmentation overlay opacity")
            min_share = gr.Slider(0.0, 5.0, value=0.1, step=0.1, label="Minimum reported cover share (%)")
            confidence = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detection confidence threshold")
            iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Detection NMS IoU threshold")
            analyze_button = gr.Button("Run complete professional assessment", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("Executive overview"):
            analysis_status = gr.Markdown()
            executive_summary = gr.HTML()
            overview_lulc = gr.HTML()
            overview_lulc_table = gr.Dataframe(
                headers=["Rank", "LULC class", "Probability (%)", "Confidence tier"],
                interactive=False,
                label="Scene classification probability profile",
            )
            with gr.Row():
                overview_segment = gr.Image(label="Pixel-level land-cover overlay")
                overview_detection = gr.Image(label="Detected objects")
            overview_files = gr.File(label="Download complete evidence package", file_count="multiple")

        with gr.Tab("LULC classification"):
            gr.Markdown("<div class='section-note'><b>Scene-level interpretation.</b> Assigns the whole image to EuroSAT land-use/land-cover classes. This is distinct from pixel segmentation and is not a legal land-use designation.</div>")
            classify_button = gr.Button("Classify scene LULC", variant="primary")
            classify_status = gr.Markdown()
            classification_assessment = gr.HTML()
            classification_table = gr.Dataframe(
                headers=["Rank", "LULC class", "Probability (%)", "Confidence tier"],
                interactive=False,
                label="Ranked LULC alternatives",
            )
            classification_files = gr.File(label="Download classification CSV / JSON", file_count="multiple")

        with gr.Tab("Land-cover segmentation"):
            gr.Markdown("<div class='section-note'><b>Pixel-level interpretation.</b> Maps nine OpenEarthMap surface classes and reports image-pixel composition.</div>")
            segment_button = gr.Button("Segment land cover", variant="primary")
            segment_status = gr.Markdown()
            with gr.Row():
                segment_overlay = gr.Image(label="Land-cover overlay")
                segment_mask = gr.Image(label="Categorical mask")
            segment_table = gr.Dataframe(
                headers=["Class ID", "Class", "Pixels", "Share (%)", "Color"],
                interactive=False,
                label="Land-cover area summary",
            )
            segment_files = gr.File(label="Download segmentation outputs", file_count="multiple")

        with gr.Tab("Object detection"):
            gr.Markdown("<div class='section-note'><b>Instance-level interpretation.</b> Locates supported objects with confidence-filtered bounding boxes.</div>")
            detect_button = gr.Button("Detect satellite objects", variant="primary")
            detect_status = gr.Markdown()
            detect_overlay = gr.Image(label="Detection overlay")
            detection_summary = gr.Dataframe(
                headers=["Class", "Count", "Average confidence", "Maximum confidence"],
                interactive=False,
                label="Detection summary",
            )
            detection_details = gr.Dataframe(
                headers=["ID", "Class", "Confidence", "x1", "y1", "x2", "y2", "Area (px²)", "Center x", "Center y"],
                interactive=False,
                label="Per-object results",
            )
            detection_files = gr.File(label="Download detection outputs", file_count="multiple")

        with gr.Tab("Methodology & scope"):
            gr.Markdown("""
### Analytical hierarchy

| Level | Question answered | Model / training domain | Output |
|---|---|---|---|
| Scene | What broad LULC type best characterizes this image? | ConvNeXT-Tiny / EuroSAT Sentinel-2 RGB | Ranked probabilities + entropy |
| Pixel | Which cover class is predicted at each pixel? | Mask2Former / OpenEarthMap | Overlay, mask, pixel shares |
| Object | Where are supported discrete objects? | YOLOv8n / NWPU VHR-10 | Boxes, counts, CSV, pixel GeoJSON |

**Interpretation guardrails:** EuroSAT is a European Sentinel-2 scene dataset; classification may shift on other sensors, regions, resolutions, or crops. Pixel shares are not automatically physical ground-area shares. Pixel-coordinate GeoJSON is not georeferenced. Models can miss small or obscured objects. Do not use outputs alone for legal, surveillance, emergency, navigation, or safety-critical decisions.
            """)

    classify_button.click(
        classify_lulc,
        inputs=[image_input, top_k],
        outputs=[classification_assessment, classification_table, classification_files, classify_status],
        api_name="classify",
    )
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
    analyze_button.click(
        analyze_satellite_image,
        inputs=[image_input, top_k, opacity, min_share, confidence, iou],
        outputs=[
            executive_summary,
            overview_lulc,
            overview_lulc_table,
            overview_segment,
            segment_mask,
            segment_table,
            overview_detection,
            detection_summary,
            detection_details,
            overview_files,
            analysis_status,
        ],
        api_name="analyze",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
