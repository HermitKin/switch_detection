from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


SPLITS = ("train", "valid", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".switch_handle_tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalize_row(raw_line: str) -> tuple[str, str]:
    values = raw_line.split()
    if not values:
        return "", "empty"

    coordinates = [float(value) for value in values[1:]]
    if len(coordinates) == 4:
        return "0 " + " ".join(values[1:]), "hbb_reclassed"

    if len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return f"0 {cx:.10f} {cy:.10f} {width:.10f} {height:.10f}", "shape_to_hbb"

    raise ValueError(f"unsupported YOLO row with {len(values)} values")


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up all labels and normalize detection classes to switch_handle.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--backup-name", default="_label_backup_before_switch_handle_20260915")
    args = parser.parse_args()

    root = args.root.resolve()
    backup_root = root / args.backup_name
    if backup_root.exists():
        raise FileExistsError(f"Refusing to overwrite label backup: {backup_root}")

    datasets = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("ds"))
    if len(datasets) != 5:
        raise RuntimeError(f"Expected 5 normalized datasets, found {len(datasets)}")

    backup_root.mkdir(parents=True)
    manifest_rows: list[dict] = []
    classification_rows: list[dict] = []
    summary: dict[str, dict] = {}

    # Complete every backup before changing any source label.
    for dataset in datasets:
        dataset_backup = backup_root / dataset.name
        stats: Counter[str] = Counter()
        for yaml_name in ("data.yaml",):
            source = dataset / yaml_name
            if source.exists():
                destination = dataset_backup / yaml_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                manifest_rows.append(
                    {
                        "dataset": dataset.name,
                        "kind": "data_yaml",
                        "source_path": str(source),
                        "backup_path": str(destination),
                        "sha256": file_sha256(source),
                    }
                )
                stats["data_yaml"] += 1

        for label_path in sorted(dataset.rglob("*.txt")):
            if label_path.parent.name != "labels" or "gt_visualization" in label_path.parts:
                continue
            relative = label_path.relative_to(dataset)
            destination = dataset_backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, destination)
            manifest_rows.append(
                {
                    "dataset": dataset.name,
                    "kind": "yolo_label",
                    "source_path": str(label_path),
                    "backup_path": str(destination),
                    "sha256": file_sha256(label_path),
                }
            )
            stats["label_files"] += 1

        for split in SPLITS:
            split_dir = dataset / split
            if not split_dir.is_dir() or (split_dir / "images").is_dir():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                        classification_rows.append(
                            {
                                "dataset": dataset.name,
                                "split": split,
                                "class": class_dir.name,
                                "image_path": str(image_path),
                            }
                        )
                        stats["classification_assignments"] += 1
        summary[dataset.name] = dict(stats)

    with (backup_root / "backup_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["dataset", "kind", "source_path", "backup_path", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (backup_root / "classification_labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "split", "class", "image_path"])
        writer.writeheader()
        writer.writerows(classification_rows)

    # Mutate only detection datasets that have data.yaml and labels directories.
    conversion_counts: dict[str, dict] = {}
    for dataset in datasets:
        yaml_path = dataset / "data.yaml"
        label_paths = sorted(
            path
            for path in dataset.rglob("*.txt")
            if path.parent.name == "labels" and "gt_visualization" not in path.parts
        )
        if not yaml_path.exists() or not label_paths:
            continue

        counts: Counter[str] = Counter()
        for label_path in label_paths:
            output_lines: list[str] = []
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    normalized, operation = normalize_row(raw_line)
                except Exception as exc:
                    raise ValueError(f"{label_path}, line {line_number}: {exc}") from exc
                output_lines.append(normalized)
                counts[operation] += 1
            atomic_write(label_path, "\n".join(output_lines) + ("\n" if output_lines else ""))
            counts["label_files"] += 1

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        config["nc"] = 1
        config["names"] = ["switch_handle"]
        atomic_write(yaml_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        counts["data_yaml_updated"] += 1
        conversion_counts[dataset.name] = dict(counts)

    result = {
        "backup_root": str(backup_root),
        "backup_manifest_entries": len(manifest_rows),
        "classification_assignments_backed_up": len(classification_rows),
        "backup_by_dataset": summary,
        "normalization": conversion_counts,
    }
    (backup_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
