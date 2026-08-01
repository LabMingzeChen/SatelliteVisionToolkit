import ast
from pathlib import Path


def test_app_exposes_all_api_endpoints_and_models():
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "segment" in constants
    assert "detect" in constants
    assert "classify" in constants
    assert "analyze" in constants
    assert "mrm8488/convnext-tiny-finetuned-eurosat" in constants
    assert "mfaytin/mask2former-satellite" in constants
    assert "bluelabel/satellite-equipment-detection-yolov8n-vhr10" in constants
    assert "cropland" in constants
    assert "residential" in constants
    assert "intersection" in constants
    assert "harbor" in constants
    assert "parking" in constants
    assert "Urban sample scenes" in source
