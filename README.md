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
short_description: LULC classification, segmentation, and object detection.
---

<div align="center">

# 🛰️ Satellite Vision Toolkit

### Professional scene, pixel, and object-level analysis for overhead imagery

[![Hugging Face Space](https://img.shields.io/badge/🤗_Hugging_Face-Live_Demo-FFD21E)](https://huggingface.co/spaces/Mingze/SatelliteVisionToolkit)
[![Detection](https://img.shields.io/badge/Detection-YOLOv8n-2563EB)](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10)
[![Segmentation](https://img.shields.io/badge/Segmentation-Mask2Former-16A34A)](https://huggingface.co/mfaytin/mask2former-satellite)
[![LULC](https://img.shields.io/badge/LULC-ConvNeXT--Tiny-0F766E)](https://huggingface.co/mrm8488/convnext-tiny-finetuned-eurosat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Upload one satellite or aerial image, run a multi-model assessment, and download reusable visual and machine-readable evidence.**

[**🚀 Launch the live app**](https://huggingface.co/spaces/Mingze/SatelliteVisionToolkit) ·
[**💻 GitHub source**](https://github.com/LabMingzeChen/SatelliteVisionToolkit)

</div>

## What it does

The app provides three complementary analytical levels plus a one-click combined workflow:

| Mode | Model | Output vocabulary |
|---|---|---|
| Scene-level LULC classification | ConvNeXT-Tiny fine-tuned on EuroSAT | annual crop, forest, herbaceous vegetation, highway, industrial, pasture, permanent crop, residential, river, sea/lake |
| Object detection | YOLOv8n fine-tuned on NWPU VHR-10 | airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle |
| Land-cover segmentation | Mask2Former fine-tuned on OpenEarthMap | background, bare land, grass, pavement, road, tree, water, cropland, building |

Every run creates visual and machine-readable outputs:

- Ranked LULC probabilities, a confidence tier, normalized entropy, and CSV/JSON exports.
- Detection overlay, per-class summary, per-object CSV, and pixel-coordinate GeoJSON.
- Land-cover overlay, categorical mask, raw class-ID PNG, and class-area CSV.
- A professional executive dashboard and complete JSON evidence package.
- Independent Gradio API endpoints at `/classify`, `/segment`, `/detect`, and `/analyze`.
- A reusable Codex skill and command-line Space client.

## Guided case studies

The interface includes three one-click NASA Earth Observatory cases. Each case loads the reference image and shows the acquisition context, recommended analytical question, and source attribution.

| Case | Sensor / date | Suggested use |
|---|---|---|
| [Indus River irrigated agriculture](https://earthobservatory.nasa.gov/images/52076/seasonal-changes-along-the-indus-river) | Landsat 5 TM / 2009-09-10 | Compare crop and river scene alternatives with cropland/water pixel shares |
| [Lluta River desert agriculture](https://earthobservatory.nasa.gov/images/82296/lluta-river-chile) | EO-1 ALI / 2012-07-19 | Inspect ambiguity where narrow irrigated valleys cross dominant bare land |
| [Zambezi wet-season floodplain](https://earthobservatory.nasa.gov/images/80835/wet-season-transforms-the-zambezi-river) | EO-1 ALI / 2013-03-31 | Evaluate water, vegetation, and bare-land composition |

These are method-exploration examples, not ground-truth demonstrations. The imagery remains credited to NASA Earth Observatory and the source instrument teams.

## How it works

```text
Satellite or aerial RGB image
    ├── ConvNeXT-Tiny / EuroSAT
    │     ├── ranked scene-level LULC probabilities
    │     ├── normalized uncertainty (entropy)
    │     └── classification CSV + JSON
    │
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

classification = client.predict(
    handle_file("satellite.jpg"),
    5,
    api_name="/classify",
)

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

complete = client.predict(
    handle_file("satellite.jpg"),
    5, 0.55, 0.10, 0.25, 0.45,
    api_name="/analyze",
)
```

The bundled CLI wraps the same endpoints:

```bash
python scripts/satellite_client.py classify satellite.jpg --output classification.json
python scripts/satellite_client.py detect satellite.jpg --output detection.json
python scripts/satellite_client.py segment satellite.jpg --output segmentation.json
python scripts/satellite_client.py analyze satellite.jpg --output complete.json
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
| [`mrm8488/convnext-tiny-finetuned-eurosat`](https://huggingface.co/mrm8488/convnext-tiny-finetuned-eurosat) | Scene-level LULC classifier | Model card lists Apache-2.0 |
| [EuroSAT](https://huggingface.co/datasets/GFM-Bench/EuroSAT) | LULC classification dataset | Review dataset terms and cite Helber et al. |
| [`bluelabel/satellite-equipment-detection-yolov8n-vhr10`](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10) | Remote-sensing object detector | Model card lists MIT; Ultralytics runtime has separate licensing |
| [NWPU VHR-10](https://gcheng-nwpu.github.io/#Datasets) | Detection training dataset | Review dataset terms and cite its authors |
| [`mfaytin/mask2former-satellite`](https://huggingface.co/mfaytin/mask2former-satellite) | Land-cover segmentation model | Model card lists MIT |
| [OpenEarthMap](https://open-earth-map.org/) | Segmentation training dataset | Review dataset terms and cite Xia et al. |

## Limitations and responsible use

- Results vary with spatial resolution, sensor, geography, season, atmospheric conditions, shadows, and image preprocessing.
- EuroSAT classification is a whole-scene hypothesis learned from small European Sentinel-2 RGB tiles; it is not parcel delineation, zoning, cadastral, or legal land-use evidence.
- Review ranked alternatives and normalized entropy. A confident prediction can still be wrong under domain shift.
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
