"""Portable train/validate/predict commands for the standard OBB baseline."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val", "predict"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="outputs/baseline")
    parser.add_argument("--name", default="switch_obb")
    args = parser.parse_args()
    if args.mode in ("train", "val") and (args.data is None or not args.data.is_file()):
        parser.error("train/val require an existing --data YAML")
    if args.mode == "predict" and (args.source is None or not args.source.exists()):
        parser.error("predict requires an existing --source")
    from ultralytics import YOLO
    model = YOLO(args.model)
    if model.task != "obb":
        raise ValueError(f"Expected an OBB model, found {model.task}")
    common = dict(device=args.device, imgsz=args.imgsz, project=args.output, name=args.name)
    if args.mode == "train":
        model.train(data=str(args.data), epochs=args.epochs, batch=args.batch, workers=args.workers,
                    seed=42, plots=True, exist_ok=False, **common)
    elif args.mode == "val":
        model.val(data=str(args.data), split=args.split, plots=True, **common)
    else:
        model.predict(source=str(args.source), save=True, **common)


if __name__ == "__main__":
    main()
