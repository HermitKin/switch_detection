from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "val", "test")


@dataclass
class ImageRecord:
    source_dataset: str
    source_split: str
    source_class: str
    original_path: str
    original_name: str
    width: int | None
    height: int | None
    image_format: str
    file_size: int
    sha256: str
    dhash: str
    label_path: str
    label_status: str
    label_rows: int
    label_formats: str
    exact_group: str
    selected_for_annotation: bool
    prepared_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:0{hash_size * hash_size // 4}x}"


def find_detection_images(dataset: Path):
    for split in SPLITS:
        image_dir = dataset / split / "images"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield split, "", image_path, dataset / split / "labels" / f"{image_path.stem}.txt"


def find_classification_images(dataset: Path):
    for split in SPLITS:
        split_dir = dataset / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    yield split, class_dir.name, image_path, None


def find_raw_images(dataset: Path):
    for image_path in sorted(dataset.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_SUFFIXES
            and "gt_visualization" not in image_path.parts
        ):
            yield "raw", "", image_path, None


def audit_label(label_path: Path | None) -> tuple[str, int, str, list[str]]:
    if label_path is None:
        return "folder_class", 0, "classification", []
    if not label_path.exists():
        return "missing", 0, "", ["missing label file"]

    issues: list[str] = []
    formats: Counter[str] = Counter()
    rows = 0
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        rows += 1
        values = raw_line.split()
        try:
            int(float(values[0]))
            coordinates = [float(value) for value in values[1:]]
        except (ValueError, IndexError):
            issues.append(f"line {line_number}: invalid numeric row")
            continue
        if len(coordinates) == 4:
            formats["hbb"] += 1
        elif len(coordinates) == 8:
            formats["obb"] += 1
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            formats["polygon"] += 1
        else:
            formats["unsupported"] += 1
            issues.append(f"line {line_number}: unsupported field count {len(values)}")
        if any(value < 0 or value > 1 for value in coordinates):
            issues.append(f"line {line_number}: coordinate outside [0, 1]")

    status = "ok" if not issues else "invalid"
    if rows == 0:
        status = "empty"
    format_text = ";".join(f"{name}:{count}" for name, count in sorted(formats.items()))
    return status, rows, format_text, issues


def safe_component(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part) or "unassigned"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit five switch datasets and build a deduplicated OBB annotation pool.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = root / "prepared_obb"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("ds"))
    if len(dataset_dirs) != 5:
        raise RuntimeError(f"Expected 5 normalized dataset directories, found {len(dataset_dirs)}")

    manifests_dir = output / "manifests"
    pool_dir = output / "annotation_pool" / "images"
    legacy_dir = output / "legacy_annotations"
    manifests_dir.mkdir(parents=True)
    pool_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)

    records: list[ImageRecord] = []
    invalid_rows: list[dict] = []
    sha_to_indices: dict[str, list[int]] = defaultdict(list)

    for dataset in dataset_dirs:
        detection_layout = any((dataset / split / "images").is_dir() for split in SPLITS)
        classification_layout = any((dataset / split).is_dir() for split in SPLITS)
        if detection_layout:
            iterator = find_detection_images(dataset)
        elif classification_layout:
            iterator = find_classification_images(dataset)
        else:
            iterator = find_raw_images(dataset)

        yaml_path = dataset / "data.yaml"
        if yaml_path.exists():
            shutil.copy2(yaml_path, legacy_dir / f"{dataset.name}_data.yaml")
        if detection_layout:
            for split in SPLITS:
                labels_dir = dataset / split / "labels"
                if labels_dir.is_dir():
                    shutil.copytree(labels_dir, legacy_dir / dataset.name / split / "labels")

        for split, source_class, image_path, label_path in iterator:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format or image_path.suffix.lstrip(".").upper()
                    dhash = difference_hash(image)
                digest = sha256_file(image_path)
                label_status, label_rows, label_formats, label_issues = audit_label(label_path)
                record = ImageRecord(
                    source_dataset=dataset.name,
                    source_split=split,
                    source_class=source_class,
                    original_path=str(image_path),
                    original_name=image_path.name,
                    width=width,
                    height=height,
                    image_format=image_format,
                    file_size=image_path.stat().st_size,
                    sha256=digest,
                    dhash=dhash,
                    label_path=str(label_path) if label_path else "",
                    label_status=label_status,
                    label_rows=label_rows,
                    label_formats=label_formats,
                    exact_group="",
                    selected_for_annotation=False,
                    prepared_name="",
                )
                index = len(records)
                records.append(record)
                sha_to_indices[digest].append(index)
                for issue in label_issues:
                    invalid_rows.append(
                        {
                            "dataset": dataset.name,
                            "image_path": str(image_path),
                            "label_path": str(label_path) if label_path else "",
                            "issue": issue,
                        }
                    )
            except Exception as exc:
                invalid_rows.append(
                    {
                        "dataset": dataset.name,
                        "image_path": str(image_path),
                        "label_path": str(label_path) if label_path else "",
                        "issue": f"image read error: {exc}",
                    }
                )

    exact_groups: list[dict] = []
    duplicate_group_number = 0
    for digest, indices in sorted(sha_to_indices.items()):
        if len(indices) <= 1:
            continue
        duplicate_group_number += 1
        group_id = f"exact_{duplicate_group_number:05d}"
        canonical = indices[0]
        for index in indices:
            records[index].exact_group = group_id
            exact_groups.append(
                {
                    "group_id": group_id,
                    "sha256": digest,
                    "canonical": index == canonical,
                    "dataset": records[index].source_dataset,
                    "split": records[index].source_split,
                    "image_path": records[index].original_path,
                }
            )

    seen_sha: set[str] = set()
    counters: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if record.sha256 in seen_sha:
            continue
        seen_sha.add(record.sha256)
        record.selected_for_annotation = True
        dataset_id = record.source_dataset.split("_", 1)[0]
        split = safe_component(record.source_split)
        source_class = safe_component(record.source_class) if record.source_class else ""
        counter_key = (dataset_id, split, source_class)
        counters[counter_key] += 1
        middle = f"_{source_class}" if source_class else ""
        extension = record.image_format.lower().replace("jpeg", "jpg")
        record.prepared_name = f"{dataset_id}_{split}{middle}_{counters[counter_key]:06d}.{extension}"
        shutil.copy2(record.original_path, pool_dir / record.prepared_name)

    near_pairs: list[dict] = []
    hash_values = [int(record.dhash, 16) for record in records]
    for left in range(len(records)):
        left_record = records[left]
        if left_record.width is None or left_record.height is None:
            continue
        left_ratio = left_record.width / left_record.height
        for right in range(left + 1, len(records)):
            right_record = records[right]
            if left_record.sha256 == right_record.sha256:
                continue
            if right_record.width is None or right_record.height is None:
                continue
            right_ratio = right_record.width / right_record.height
            if abs(left_ratio - right_ratio) / max(left_ratio, right_ratio) > 0.02:
                continue
            distance = (hash_values[left] ^ hash_values[right]).bit_count()
            if distance <= 4:
                near_pairs.append(
                    {
                        "dhash_distance": distance,
                        "left_dataset": left_record.source_dataset,
                        "left_split": left_record.source_split,
                        "left_path": left_record.original_path,
                        "right_dataset": right_record.source_dataset,
                        "right_split": right_record.source_split,
                        "right_path": right_record.original_path,
                    }
                )

    manifest_rows = [asdict(record) for record in records]
    write_csv(manifests_dir / "dataset_manifest.csv", manifest_rows, list(asdict(records[0]).keys()))
    write_csv(
        manifests_dir / "exact_duplicates.csv",
        exact_groups,
        ["group_id", "sha256", "canonical", "dataset", "split", "image_path"],
    )
    write_csv(
        manifests_dir / "near_duplicate_pairs.csv",
        near_pairs,
        [
            "dhash_distance",
            "left_dataset",
            "left_split",
            "left_path",
            "right_dataset",
            "right_split",
            "right_path",
        ],
    )
    write_csv(
        manifests_dir / "invalid_images_or_labels.csv",
        invalid_rows,
        ["dataset", "image_path", "label_path", "issue"],
    )

    summary = {
        "datasets": {dataset.name: sum(record.source_dataset == dataset.name for record in records) for dataset in dataset_dirs},
        "total_valid_images": len(records),
        "unique_exact_images": len(seen_sha),
        "exact_duplicate_images_removed_from_pool": len(records) - len(seen_sha),
        "exact_duplicate_groups": duplicate_group_number,
        "near_duplicate_candidate_pairs": len(near_pairs),
        "invalid_image_or_label_issues": len(invalid_rows),
        "annotation_pool": str(pool_dir),
    }
    (manifests_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
