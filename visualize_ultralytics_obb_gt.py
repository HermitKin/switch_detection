from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = ((255, 70, 70), (40, 170, 255), (70, 210, 110), (255, 180, 40))


def load_font(size: int):
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_names(dataset: Path) -> list[str]:
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def draw_label(draw: ImageDraw.ImageDraw, point: tuple[float, float], text: str, color, font) -> None:
    x, y = point
    box = draw.textbbox((x, y), text, font=font)
    pad = 4
    width = box[2] - box[0] + pad * 2
    height = box[3] - box[1] + pad * 2
    y = max(0, y - height)
    draw.rectangle((x, y, x + width, y + height), fill=color)
    draw.text((x + pad, y + pad), text, fill="white", font=font)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    names = load_names(dataset)
    output = dataset / "gt_visualization"
    counts: Counter[str] = Counter()
    errors: list[str] = []

    for split in ("train", "valid", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir():
            image_dir = dataset / split / "images"
            label_dir = dataset / split / "labels"
        if not image_dir.is_dir():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            destination = output / split / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                width, height = image.size
                draw = ImageDraw.Draw(image, "RGBA")
                thickness = max(3, round(min(width, height) / 250))
                font = load_font(max(15, round(min(width, height) / 30)))
                if not label_path.exists():
                    raise FileNotFoundError(f"missing label: {label_path}")

                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    values = line.split()
                    if len(values) != 9:
                        raise ValueError(f"line {line_number}: expected 9 values, got {len(values)}")
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                    points = [
                        (coordinates[index] * width, coordinates[index + 1] * height)
                        for index in range(0, 8, 2)
                    ]
                    color = COLORS[class_id % len(COLORS)]
                    draw.polygon(points, fill=(*color, 38))
                    draw.line(points + [points[0]], fill=(*color, 255), width=thickness, joint="curve")
                    # White first edge makes the stored OBB vertex order visible.
                    draw.line((points[0], points[1]), fill=(255, 255, 255, 255), width=max(1, thickness // 2))
                    name = names[class_id] if 0 <= class_id < len(names) else "unknown"
                    draw_label(draw, points[0], f"{class_id}:{name}", color, font)
                    counts[f"{class_id}:{name}"] += 1

                image.save(destination, quality=92)
                counts["images"] += 1
            except Exception as exc:
                errors.append(f"{image_path}: {exc}")

    summary = {
        "dataset": str(dataset),
        "output": str(output),
        "format": "Ultralytics YOLO OBB",
        "counts": dict(counts),
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
