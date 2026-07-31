#!/usr/bin/env python3
"""Call the public Satellite Vision Toolkit Space from a terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradio_client import Client, handle_file


DEFAULT_SPACE = "Mingze/SatelliteVisionToolkit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("detect", "segment"))
    parser.add_argument("image", type=Path)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--min-share", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    client = Client(args.space)
    if args.operation == "detect":
        result = client.predict(
            handle_file(str(args.image)),
            args.confidence,
            args.iou,
            api_name="/detect",
        )
    else:
        result = client.predict(
            handle_file(str(args.image)),
            args.opacity,
            args.min_share,
            api_name="/segment",
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
