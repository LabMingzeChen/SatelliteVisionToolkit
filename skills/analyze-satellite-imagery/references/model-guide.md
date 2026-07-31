# Model and output guide

## Scene-level LULC classification

- Model: `mrm8488/convnext-tiny-finetuned-eurosat`
- Architecture: ConvNeXT-Tiny
- Training data: EuroSAT RGB images derived from European Sentinel-2 imagery
- Classes: annual crop, forest, herbaceous vegetation, highway, industrial, pasture, permanent crop, residential, river, sea/lake
- Model-card license: Apache-2.0
- Model-card reported evaluation accuracy: 0.9805

The classifier assigns one broad label to the processed image. It does not delineate parcels or pixels and must not be represented as a cadastral, zoning, or legal land-use conclusion. EuroSAT images are small European Sentinel-2 tiles; other sensors, countries, seasons, scales, and image crops introduce domain shift. Review the full probability profile and normalized entropy, not only the top label.

## Object detection

- Model: `bluelabel/satellite-equipment-detection-yolov8n-vhr10`
- Architecture: YOLOv8n
- Training data: NWPU VHR-10
- Classes: airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle
- Weights license: MIT according to the model card
- Runtime: Ultralytics; review its current licensing before redistribution or commercial deployment

Small objects are especially sensitive to image resizing and ground resolution. A missing box does not establish absence. The model card reports aggregate validation metrics, which do not guarantee performance on a new region or sensor.

## Land-cover segmentation

- Model: `mfaytin/mask2former-satellite`
- Architecture: Mask2Former with a Swin backbone
- Training data: OpenEarthMap
- Classes: background, bare land, grass, pavement, road, tree, water, cropland, building
- Model-card license: MIT
- Reported best validation mIoU: 0.5202

Area share is computed as `class pixels / all processed pixels`. It is a two-dimensional image fraction and is not automatically a ground-area fraction when imagery is oblique, warped, or unreferenced.

## Export semantics

- Classification CSV and JSON preserve ranked probabilities, confidence tiers, and normalized entropy.
- Complete-analysis JSON records all three model IDs, summaries, thresholds-derived outputs, timing, and coordinate caveats.
- Segmentation class-ID PNG preserves numeric predicted labels.
- Segmentation CSV contains class ID, name, pixel count, share percent, and display color.
- Detection CSV contains image-pixel boxes, pixel area, and normalized box centers.
- Detection GeoJSON stores box polygons in image-pixel coordinates with origin at the top left. It is intentionally not assigned a geographic coordinate reference system.
