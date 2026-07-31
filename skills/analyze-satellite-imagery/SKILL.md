---
name: analyze-satellite-imagery
description: Classify land use and land cover, segment surface classes, and detect remote-sensing objects in satellite or aerial RGB imagery. Use when Codex needs to analyze overhead PNG, JPEG, WebP, or TIFF images; assign EuroSAT LULC scene classes; locate airplanes, ships, vehicles, storage tanks, bridges, harbors, or sports facilities; map OpenEarthMap land-cover classes; produce professional summaries, overlays, masks, CSV, JSON, or pixel-coordinate GeoJSON; or call the Satellite Vision Toolkit Hugging Face Space API.
---

# Analyze Satellite Imagery

Use the bundled Space or local app to run three complementary analytical levels:

- Classify the whole scene across 10 EuroSAT LULC categories with ConvNeXT-Tiny.
- Detect 10 NWPU VHR-10 object categories with the fine-tuned YOLOv8n model.
- Segment 9 OpenEarthMap land-cover categories with Mask2Former.

## Choose a workflow

1. Use classification for a broad whole-scene LULC hypothesis and ranked alternatives.
2. Use segmentation for per-pixel land-cover composition.
3. Use detection for discrete objects and bounding boxes.
4. Use complete analysis when a professional summary or cross-method evidence package is needed.
5. Inspect `references/model-guide.md` before making claims about model scope, licenses, or limitations.

## Run the app

From the plugin root, install `requirements.txt` and run `python app.py`. For a deployed Space, use the browser UI or call `/classify`, `/segment`, `/detect`, or `/analyze` with `gradio_client`.

Use `scripts/satellite_client.py` for repeatable API calls:

```bash
python scripts/satellite_client.py classify image.jpg --output classification.json
python scripts/satellite_client.py detect image.jpg --output result.json
python scripts/satellite_client.py segment image.jpg --output result.json
python scripts/satellite_client.py analyze image.jpg --output complete.json
```

Set `--space` when using a fork. The default is `Mingze/SatelliteVisionToolkit`.

## Interpret outputs

- Keep scene-level classification separate from pixel-level segmentation in the report.
- Treat EuroSAT probabilities as a broad scene hypothesis, not zoning, cadastral, or legal land-use evidence.
- Report normalized entropy when classification ambiguity matters; a high top score does not remove domain-shift risk.
- Treat detection counts as visible-image estimates, not inventories.
- Treat class shares as proportions of processed image pixels, not physical land area.
- State that exported GeoJSON uses top-left-origin image pixels and has no geographic CRS.
- Do not infer ground distance or area without image georeferencing and ground sample distance.
- Flag cloud, haze, shadows, seasonal change, resolution mismatch, and sensor mismatch when relevant.
- Avoid safety-critical, legal-boundary, surveillance, military targeting, or emergency-response conclusions.

## Report results

Include the analytical level, model, thresholds, processed image dimensions, ranked LULC alternatives, detected classes/counts or land-cover shares, and important limitations. Link or return generated overlays and exports when available.
