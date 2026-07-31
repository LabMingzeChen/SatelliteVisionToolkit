---
name: analyze-satellite-imagery
description: Detect remote-sensing objects and segment land cover in satellite or aerial RGB imagery. Use when Codex needs to analyze overhead PNG, JPEG, WebP, or TIFF images; locate airplanes, ships, vehicles, storage tanks, bridges, harbors, or sports facilities; map OpenEarthMap land-cover classes; produce overlays, masks, CSV summaries, or pixel-coordinate GeoJSON; or call the Satellite Vision Toolkit Hugging Face Space API.
---

# Analyze Satellite Imagery

Use the bundled Space or local app to run two complementary workflows:

- Detect 10 NWPU VHR-10 object categories with the fine-tuned YOLOv8n model.
- Segment 9 OpenEarthMap land-cover categories with Mask2Former.

## Choose a workflow

1. Use detection for discrete objects and bounding boxes.
2. Use segmentation for per-pixel land-cover composition.
3. Run both when the question mixes infrastructure counts and surface coverage.
4. Inspect `references/model-guide.md` before making claims about model scope, licenses, or limitations.

## Run the app

From the plugin root, install `requirements.txt` and run `python app.py`. For a deployed Space, use the browser UI or call the `/detect` and `/segment` endpoints with `gradio_client`.

Use `scripts/satellite_client.py` for repeatable API calls:

```bash
python scripts/satellite_client.py detect image.jpg --output result.json
python scripts/satellite_client.py segment image.jpg --output result.json
```

Set `--space` when using a fork. The default is `Mingze/SatelliteVisionToolkit`.

## Interpret outputs

- Treat detection counts as visible-image estimates, not inventories.
- Treat class shares as proportions of processed image pixels, not physical land area.
- State that exported GeoJSON uses top-left-origin image pixels and has no geographic CRS.
- Do not infer ground distance or area without image georeferencing and ground sample distance.
- Flag cloud, haze, shadows, seasonal change, resolution mismatch, and sensor mismatch when relevant.
- Avoid safety-critical, legal-boundary, surveillance, military targeting, or emergency-response conclusions.

## Report results

Include the model, thresholds, processed image dimensions, detected classes/counts or land-cover shares, and the important limitations. Link or return the generated overlays and exports when available.
