"""Read-only checks for a two-class Ultralytics OBB dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "switch_handle", 1: "angle"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )) / 2


def audit(root: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    per_split = {}
    hashes: dict[str, list[str]] = defaultdict(list)
    all_image_names: dict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images/ or labels/ directory")
            continue

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            errors.append(f"{split}: image has no label: {stem}")
        for stem in sorted(label_stems - image_stems):
            errors.append(f"{split}: orphan label: {stem}")

        class_counts: Counter[int] = Counter()
        empty_labels = 0
        per_image_mismatches = 0
        for image in images:
            hashes[hash_file(image)].append(f"{split}/{image.name}")
            all_image_names[image.name].append(split)
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            counts: Counter[int] = Counter()
            lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                location = f"{split}/{label.name}:{line_number}"
                if len(fields) != 9:
                    errors.append(f"{location}: expected 9 fields, got {len(fields)}")
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{location}: non-numeric value")
                    continue
                if class_id not in CLASS_NAMES:
                    errors.append(f"{location}: unsupported class {class_id}")
                    continue
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                    errors.append(f"{location}: coordinates outside [0, 1] or non-finite")
                    continue
                points = list(zip(coordinates[::2], coordinates[1::2]))
                if polygon_area(points) <= 1e-8:
                    errors.append(f"{location}: degenerate OBB polygon")
                    continue
                class_counts[class_id] += 1
                counts[class_id] += 1
            if counts[0] != counts[1]:
                per_image_mismatches += 1
                warnings.append(
                    f"{split}/{image.name}: switch_handle={counts[0]}, angle={counts[1]}"
                )

        per_split[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": {CLASS_NAMES[key]: class_counts[key] for key in CLASS_NAMES},
            "empty_labels": empty_labels,
            "class_count_mismatch_images": per_image_mismatches,
        }

    duplicate_content = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_names = {name: splits for name, splits in all_image_names.items() if len(splits) > 1}
    if duplicate_content:
        warnings.append(f"exact duplicate images: {len(duplicate_content)} groups")
    if duplicate_names:
        warnings.append(f"same image filename across splits: {len(duplicate_names)} names")

    return {
        "root": str(root),
        "expected_classes": CLASS_NAMES,
        "splits": per_split,
        "exact_duplicate_image_groups": duplicate_content[:20],
        "duplicate_image_names": dict(list(duplicate_names.items())[:20]),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
        "ready_for_training": len(errors) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
