from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "val", "test")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def class_names(dataset_dir: Path) -> list[str]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return []
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return [str(name) for name in names]


def color_for(class_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 82, 82),
        (0, 200, 255),
        (60, 210, 120),
        (255, 180, 0),
        (190, 100, 255),
        (0, 220, 210),
        (255, 100, 180),
    )
    return palette[class_id % len(palette)]


def draw_tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font, stroke_width=0)
    pad = max(2, int(getattr(font, "size", 12) * 0.2))
    width = box[2] - box[0] + pad * 2
    height = box[3] - box[1] + pad * 2
    y = max(0, y - height)
    draw.rectangle((x, y, x + width, y + height), fill=fill)
    draw.text((x + pad, y + pad), text, font=font, fill=(255, 255, 255))


def visualize_detection_dataset(dataset_dir: Path, output_dir: Path) -> dict:
    names = class_names(dataset_dir)
    counts: Counter[str] = Counter()
    errors: list[str] = []

    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.is_dir():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = Path(split) / image_path.name
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                width, height = image.size
                thickness = max(2, round(min(width, height) / 300))
                font = load_font(max(12, round(min(width, height) / 35)))
                draw = ImageDraw.Draw(image)
                object_count = 0

                lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
                for line_number, raw_line in enumerate(lines, start=1):
                    if not raw_line.strip():
                        continue
                    values = raw_line.split()
                    try:
                        class_id = int(float(values[0]))
                        coordinates = [float(value) for value in values[1:]]
                    except ValueError as exc:
                        raise ValueError(f"line {line_number}: invalid numeric value") from exc

                    class_name = names[class_id] if 0 <= class_id < len(names) else "unknown"
                    label = f"{class_id}:{class_name}"
                    color = color_for(class_id)

                    if len(coordinates) == 4:
                        cx, cy, box_w, box_h = coordinates
                        x1 = (cx - box_w / 2) * width
                        y1 = (cy - box_h / 2) * height
                        x2 = (cx + box_w / 2) * width
                        y2 = (cy + box_h / 2) * height
                        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
                        draw_tag(draw, (max(0, x1), max(0, y1)), label, font, color)
                    elif len(coordinates) == 8:
                        points = [
                            (coordinates[index] * width, coordinates[index + 1] * height)
                            for index in range(0, 8, 2)
                        ]
                        draw.line(points + [points[0]], fill=color, width=thickness, joint="curve")
                        draw_tag(draw, points[0], label, font, color)
                        # Highlight the first annotated edge so OBB vertex order is visible.
                        draw.line((points[0], points[1]), fill=(255, 255, 255), width=max(1, thickness // 2))
                    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                        points = [
                            (coordinates[index] * width, coordinates[index + 1] * height)
                            for index in range(0, len(coordinates), 2)
                        ]
                        draw.line(points + [points[0]], fill=color, width=thickness, joint="curve")
                        draw_tag(draw, points[0], f"{label} polygon", font, color)
                    else:
                        raise ValueError(
                            f"line {line_number}: unsupported YOLO row with {len(values)} values"
                        )

                    counts[label] += 1
                    object_count += 1

                if object_count == 0:
                    status = "NO GT OBJECTS" if label_path.exists() else "MISSING LABEL FILE"
                    draw_tag(draw, (8, max(24, getattr(font, "size", 12) + 8)), status, font, (220, 60, 60))
                    counts[status] += 1

                image.save(destination, quality=92)
                counts["images"] += 1
            except Exception as exc:  # Continue the batch and record the exact failing image.
                errors.append(f"{image_path}: {exc}")

    return {"type": "detection", "counts": dict(counts), "errors": errors}


def visualize_classification_dataset(dataset_dir: Path, output_dir: Path) -> dict:
    counts: Counter[str] = Counter()
    errors: list[str] = []

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                destination = output_dir / split / class_dir.name / image_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    width, height = image.size
                    font = load_font(max(14, round(min(width, height) / 28)))
                    draw = ImageDraw.Draw(image)
                    color = color_for(abs(hash(class_dir.name)) % 7)
                    draw_tag(draw, (8, max(28, getattr(font, "size", 14) + 10)), f"GT: {class_dir.name}", font, color)
                    image.save(destination, quality=92)
                    counts["images"] += 1
                    counts[class_dir.name] += 1
                except Exception as exc:
                    errors.append(f"{image_path}: {exc}")

    return {"type": "classification", "counts": dict(counts), "errors": errors}


def visualize_dataset(dataset_dir: Path) -> dict:
    output_dir = dataset_dir / "gt_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    has_detection_layout = any((dataset_dir / split / "images").is_dir() for split in SPLITS)
    if has_detection_layout:
        summary = visualize_detection_dataset(dataset_dir, output_dir)
    else:
        summary = visualize_classification_dataset(dataset_dir, output_dir)

    summary["dataset"] = dataset_dir.name
    summary["output"] = str(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GT labels for YOLO detection/OBB or folder classification datasets.")
    parser.add_argument("root", type=Path, help="Directory containing one or more dataset directories")
    args = parser.parse_args()

    root = args.root.resolve()
    datasets = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("ds"))
    summaries = [visualize_dataset(dataset) for dataset in datasets]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 1 if any(summary["errors"] for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
