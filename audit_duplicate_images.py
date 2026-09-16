from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集")
DATASETS = (
    "ds01_switch_state_det_4cls",
    "ds02_rotary_switch_det_1cls",
    "ds03_rotary_switch_raw",
    "ds04_switch_state_det_3cls",
    "ds05_switch_state_cls_3cls",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_PRIORITY = {name: index for index, name in enumerate(DATASETS)}
REPORT_CSV = ROOT / "duplicate_images_sha256_report.csv"
SUMMARY_JSON = ROOT / "duplicate_images_sha256_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(image: Path, dataset: Path) -> Path | None:
    relative = image.relative_to(dataset)
    parts = relative.parts
    if len(parts) >= 3 and parts[1] == "images" and parts[0] in {"train", "valid", "val", "test"}:
        candidate = dataset / parts[0] / "labels" / f"{image.stem}.txt"
        return candidate if candidate.exists() else None
    return None


def location_fields(image: Path, dataset: Path) -> tuple[str, str]:
    relative = image.relative_to(dataset)
    parts = relative.parts
    split = parts[0] if parts and parts[0] in {"train", "valid", "val", "test"} else "raw"
    category = ""
    if len(parts) >= 2 and parts[0] in {"train", "valid", "val", "test"} and parts[1] != "images":
        category = parts[1]
    return split, category


def main() -> int:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    total_images = 0

    for dataset_name in DATASETS:
        dataset = ROOT / dataset_name
        for image in sorted(dataset.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image.relative_to(dataset)
            if relative.parts and relative.parts[0] == "gt_visualization":
                continue
            label = label_for(image, dataset)
            split, category = location_fields(image, dataset)
            label_hash = sha256(label) if label else ""
            object_count = 0
            if label:
                object_count = sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
            image_hash = sha256(image)
            by_hash[image_hash].append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "category": category,
                    "image": image,
                    "label": label,
                    "label_sha256": label_hash,
                    "object_count": object_count,
                    "bytes": image.stat().st_size,
                }
            )
            total_images += 1

    duplicate_groups = [(digest, items) for digest, items in by_hash.items() if len(items) > 1]
    duplicate_groups.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    rows: list[dict[str, object]] = []
    conflict_groups = 0
    duplicate_instances = 0

    for group_number, (digest, items) in enumerate(duplicate_groups, start=1):
        label_hashes = {str(item["label_sha256"]) for item in items if item["label_sha256"]}
        label_conflict = len(label_hashes) > 1
        if label_conflict:
            conflict_groups += 1

        # Prefer a labeled copy. Then use stable dataset priority and path ordering.
        ordered = sorted(
            items,
            key=lambda item: (
                0 if item["label"] else 1,
                DATASET_PRIORITY[str(item["dataset"])],
                str(item["image"]).casefold(),
            ),
        )
        keeper = ordered[0]
        duplicate_instances += len(items) - 1

        for item in ordered:
            is_keeper = item is keeper
            if is_keeper:
                recommendation = "KEEP"
            elif label_conflict:
                recommendation = "REVIEW_LABEL_CONFLICT"
            else:
                recommendation = "DELETE_CANDIDATE"
            image = item["image"]
            label = item["label"]
            rows.append(
                {
                    "group_id": f"dup_{group_number:04d}",
                    "sha256": digest,
                    "copies": len(items),
                    "recommendation": recommendation,
                    "label_conflict": str(label_conflict).lower(),
                    "dataset": item["dataset"],
                    "split": item["split"],
                    "category": item["category"],
                    "image_bytes": item["bytes"],
                    "object_count": item["object_count"],
                    "image_path": str(image),
                    "label_path": str(label) if label else "",
                    "label_sha256": item["label_sha256"],
                }
            )

    fieldnames = [
        "group_id",
        "sha256",
        "copies",
        "recommendation",
        "label_conflict",
        "dataset",
        "split",
        "category",
        "image_bytes",
        "object_count",
        "image_path",
        "label_path",
        "label_sha256",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "root": str(ROOT),
        "hash_algorithm": "SHA-256",
        "scope": list(DATASETS),
        "excluded": ["gt_visualization", "_label_backup_before_switch_handle_20260915"],
        "total_source_images": total_images,
        "unique_image_hashes": len(by_hash),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_instances_beyond_one_keeper": duplicate_instances,
        "label_conflict_groups": conflict_groups,
        "report_csv": str(REPORT_CSV),
        "no_files_deleted": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
