---
title: Satellite Vision Toolkit
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Satellite object detection and land-cover segmentation.
---

<div align="center">

# 🛰️ Satellite Vision Toolkit

### Object detection and pixel-level land-cover mapping for overhead imagery

[![Hugging Face Space](https://img.shields.io/badge/🤗_Hugging_Face-Live_Demo-FFD21E)](https://huggingface.co/spaces/Mingze/SatelliteVisionToolkit)
[![Detection](https://img.shields.io/badge/Detection-YOLOv8n-2563EB)](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10)
[![Segmentation](https://img.shields.io/badge/Segmentation-Mask2Former-16A34A)](https://huggingface.co/mfaytin/mask2former-satellite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Upload one satellite or aerial image, detect remote-sensing objects, map land cover, and download reusable evidence.**

[**🚀 Launch the live app**](https://huggingface.co/spaces/Mingze/SatelliteVisionToolkit) ·
[**💻 GitHub source**](https://github.com/LabMingzeChen/SatelliteVisionToolkit)

</div>

## What it does

The app provides two complementary analysis modes:

| Mode | Model | Output vocabulary |
|---|---|---|
| Object detection | YOLOv8n fine-tuned on NWPU VHR-10 | airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle |
| Land-cover segmentation | Mask2Former fine-tuned on OpenEarthMap | background, bare land, grass, pavement, road, tree, water, cropland, building |

Every run creates visual and machine-readable outputs:

- Detection overlay, per-class summary, per-object CSV, and pixel-coordinate GeoJSON.
- Land-cover overlay, categorical mask, raw class-ID PNG, and class-area CSV.
- Independent Gradio API endpoints at `/detect` and `/segment`.
- A reusable Codex skill and command-line Space client.

## How it works

```text
Satellite or aerial RGB image
    ├── YOLOv8n / NWPU VHR-10
    │     ├── labeled bounding-box overlay
    │     ├── class counts and confidence
    │     ├── per-object CSV
    │     └── image-pixel GeoJSON
    │
    └── Mask2Former / OpenEarthMap
          ├── land-cover overlay
          ├── categorical color mask
          ├── raw class-ID PNG
          └── per-class pixel-share CSV
```

Images are orientation-corrected, converted to RGB, and bounded to 2048 pixels on their longest side. The first request downloads the public model weights; later requests reuse the container cache. CUDA is used when available and CPU remains supported.

## Run locally

```bash
git clone https://github.com/LabMingzeChen/SatelliteVisionToolkit.git
cd SatelliteVisionToolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL. PNG, JPEG, WebP, and RGB TIFF inputs work best.

## Call the Hugging Face API

```python
from gradio_client import Client, handle_file

client = Client("Mingze/SatelliteVisionToolkit")

detection = client.predict(
    handle_file("satellite.jpg"),
    0.25,
    0.45,
    api_name="/detect",
)

segmentation = client.predict(
    handle_file("satellite.jpg"),
    0.55,
    0.10,
    api_name="/segment",
)
```

The bundled CLI wraps the same endpoints:

```bash
python scripts/satellite_client.py detect satellite.jpg --output detection.json
python scripts/satellite_client.py segment satellite.jpg --output segmentation.json
```

## Project structure

```text
SatelliteVisionToolkit/
├── app.py                     Gradio UI and inference workflows
├── satellite_utils.py         Rendering, summaries, CSV, and GeoJSON exports
├── scripts/satellite_client.py
├── tests/                     Lightweight deterministic tests
├── skills/                    Reusable Codex workflow
└── .codex-plugin/plugin.json  Codex plugin manifest
```

## Models, data, and licensing

The application code is MIT licensed. Model software, weights, and training data keep their own terms.

| Resource | Role | Terms noted by source |
|---|---|---|
| [`bluelabel/satellite-equipment-detection-yolov8n-vhr10`](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10) | Remote-sensing object detector | Model card lists MIT; Ultralytics runtime has separate licensing |
| [NWPU VHR-10](https://gcheng-nwpu.github.io/#Datasets) | Detection training dataset | Review dataset terms and cite its authors |
| [`mfaytin/mask2former-satellite`](https://huggingface.co/mfaytin/mask2former-satellite) | Land-cover segmentation model | Model card lists MIT |
| [OpenEarthMap](https://open-earth-map.org/) | Segmentation training dataset | Review dataset terms and cite Xia et al. |

## Limitations and responsible use

- Results vary with spatial resolution, sensor, geography, season, atmospheric conditions, shadows, and image preprocessing.
- Small objects may disappear during resizing or fall below the confidence threshold.
- Detection counts describe visible predictions, not complete inventories.
- Segmentation shares describe processed image pixels, not surveyed ground area.
- Exported GeoJSON uses top-left-origin image pixels and has no geographic CRS. It must not be overlaid on a map as if it were georeferenced.
- Do not use predictions alone for navigation, legal boundaries, surveillance, military targeting, emergency response, or other safety-critical decisions.
- Avoid uploading private or sensitive imagery to a public Space.

## Citation

```bibtex
@software{chen2026satellitevisiontoolkit,
  author = {Chen, Mingze},
  title  = {Satellite Vision Toolkit},
  year   = {2026},
  url    = {https://github.com/LabMingzeChen/SatelliteVisionToolkit}
}
```

Please also cite NWPU VHR-10, OpenEarthMap, YOLO/Ultralytics, and Mask2Former as applicable.
