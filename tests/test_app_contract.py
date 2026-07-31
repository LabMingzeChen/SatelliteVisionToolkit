import ast
from pathlib import Path


def test_app_exposes_both_api_endpoints():
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "segment" in constants
    assert "detect" in constants
    assert "mfaytin/mask2former-satellite" in constants
    assert "bluelabel/satellite-equipment-detection-yolov8n-vhr10" in constants
    assert "cropland" in constants
