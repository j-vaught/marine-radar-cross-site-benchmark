"""Evaluate stored COCO predictions against an explicit reference subset."""

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_coco
from .protocol import save_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictions = json.loads(args.predictions.read_text())
    if isinstance(predictions, dict):
        predictions = predictions["predictions"]
    result = evaluate_coco(json.loads(args.annotations.read_text()), predictions)
    save_json(
        args.output,
        {
            "metrics": result,
            "annotation_sha256": sha256_file(args.annotations),
            "predictions_sha256": sha256_file(args.predictions),
        },
    )


if __name__ == "__main__":
    main()
