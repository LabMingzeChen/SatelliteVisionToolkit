import json

import numpy as np
from PIL import Image

import satellite_utils as utils


DETECTIONS = [
    {
        "class_id": 0,
        "class_name": "airplane",
        "confidence": 0.9,
        "x1": 10.0,
        "y1": 20.0,
        "x2": 30.0,
        "y2": 40.0,
    },
    {
        "class_id": 9,
        "class_name": "vehicle",
        "confidence": 0.7,
        "x1": 50.0,
        "y1": 10.0,
        "x2": 60.0,
        "y2": 20.0,
    },
    {
        "class_id": 9,
        "class_name": "vehicle",
        "confidence": 0.5,
        "x1": 70.0,
        "y1": 30.0,
        "x2": 80.0,
        "y2": 40.0,
    },
]


def test_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (4000, 2000), "white")
    assert utils.resize_for_inference(image, max_side=1000).size == (1000, 500)


def test_class_table_is_sorted_and_thresholded():
    class_map = np.array([[4, 4], [4, 6]], dtype=np.uint8)
    rows = utils.build_class_table(
        class_map,
        {4: "road", 6: "water"},
        min_share_percent=30.0,
    )
    assert rows == [[4, "road", 3, 75.0, "#5C5C5C"]]


def test_segmentation_outputs_match_input_size():
    image = Image.new("RGB", (3, 2), "black")
    class_map = np.array([[4, 4, 6], [4, 6, 6]], dtype=np.uint8)
    overlay, mask = utils.render_segmentation(
        image,
        class_map,
        {4: "road", 6: "water"},
        0.5,
    )
    assert overlay.size == image.size
    assert mask.size == image.size


def test_detection_summary_and_normalized_centers():
    assert utils.build_detection_summary(DETECTIONS) == [
        ["vehicle", 2, 0.6, 0.7],
        ["airplane", 1, 0.9, 0.9],
    ]
    rows = utils.build_detection_table(DETECTIONS[:1], (100, 100))
    assert rows[0][-2:] == [0.2, 0.3]


def test_pixel_geojson_is_explicitly_unreferenced(tmp_path):
    path = tmp_path / "detections.geojson"
    utils.write_pixel_geojson(path, DETECTIONS[:1], (100, 80))
    data = json.loads(path.read_text())
    assert data["properties"]["coordinate_system"] == "image_pixels"
    assert data["properties"]["origin"] == "top_left"
    assert data["features"][0]["geometry"]["coordinates"][0][0] == [10.0, 20.0]
