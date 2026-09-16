from __future__ import annotations

import csv
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class RenameOp:
    source: Path
    target: Path


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.casefold(),
    )


def add_group(
    dataset: Path,
    dataset_code: str,
    group_code: str,
    image_dir: Path,
    label_dir: Path | None,
    visual_dir: Path | None,
    rows: list[dict[str, str]],
    ops: list[RenameOp],
) -> None:
    images = image_files(image_dir)
    for index, image_path in enumerate(images, start=1):
        new_stem = f"{dataset_code}_{group_code}_{index:06d}"
        new_image = image_path.with_name(new_stem + image_path.suffix.lower())
        label_path = label_dir / f"{image_path.stem}.txt" if label_dir else None
        new_label = label_dir / f"{new_stem}.txt" if label_dir else None
        visual_path = visual_dir / image_path.name if visual_dir else None
        new_visual = visual_dir / new_image.name if visual_dir else None

        if label_path is not None and not label_path.exists():
            raise RuntimeError(f"Missing matching label: {label_path}")

        ops.append(RenameOp(image_path, new_image))
        if label_path is not None and new_label is not None:
            ops.append(RenameOp(label_path, new_label))
        if visual_path is not None and new_visual is not None and visual_path.exists():
            ops.append(RenameOp(visual_path, new_visual))

        rows.append(
            {
                "dataset": dataset.name,
                "group": group_code,
                "old_image": str(image_path.relative_to(dataset)),
                "new_image": str(new_image.relative_to(dataset)),
                "old_label": str(label_path.relative_to(dataset)) if label_path else "",
                "new_label": str(new_label.relative_to(dataset)) if new_label else "",
            }
        )


def validate_ops(ops: list[RenameOp]) -> None:
    sources = {op.source.resolve() for op in ops}
    targets = [op.target.resolve() for op in ops]
    if len(targets) != len(set(targets)):
        raise RuntimeError("Duplicate rename targets detected")
    for op in ops:
        if not op.source.exists():
            raise RuntimeError(f"Rename source does not exist: {op.source}")
        if op.target.resolve() not in sources and op.target.exists():
            raise RuntimeError(f"Rename target already exists: {op.target}")


def apply_ops(ops: list[RenameOp]) -> None:
    staged: list[tuple[Path, Path]] = []
    for op in ops:
        temporary = op.source.with_name(f".__rename_{uuid.uuid4().hex}{op.source.suffix}")
        op.source.rename(temporary)
        staged.append((temporary, op.target))
    for temporary, target in staged:
        temporary.rename(target)


def assert_detection_pairs(dataset: Path) -> None:
    for split in SPLITS:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.is_dir():
            continue
        image_stems = {p.stem for p in image_files(image_dir)}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            missing = sorted(image_stems - label_stems)[:5]
            orphan = sorted(label_stems - image_stems)[:5]
            raise RuntimeError(f"Pair check failed for {dataset.name}/{split}: missing={missing}, orphan={orphan}")


def main() -> None:
    specifications = {
        "ds01_switch_state_det_4cls": "ds01",
        "ds02_rotary_switch_det_1cls": "ds02",
        "ds04_switch_state_det_3cls": "ds04",
    }

    total_rows = 0
    for dataset_name, dataset_code in specifications.items():
        dataset = ROOT / dataset_name
        manifest = dataset / "rename_manifest.csv"
        if manifest.exists():
            print(f"SKIP {dataset_name}: rename_manifest.csv already exists")
            continue
        rows: list[dict[str, str]] = []
        ops: list[RenameOp] = []
        for split in SPLITS:
            image_dir = dataset / split / "images"
            if image_dir.is_dir():
                add_group(
                    dataset,
                    dataset_code,
                    split,
                    image_dir,
                    dataset / split / "labels",
                    dataset / "gt_visualization" / split,
                    rows,
                    ops,
                )
        validate_ops(ops)
        apply_ops(ops)
        with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        assert_detection_pairs(dataset)
        total_rows += len(rows)
        print(f"RENAMED {dataset_name}: {len(rows)} images, matching labels and GT previews")

    dataset = ROOT / "ds03_rotary_switch_raw"
    manifest = dataset / "rename_manifest.csv"
    if not manifest.exists():
        rows = []
        ops = []
        add_group(
            dataset,
            "ds03",
            "raw",
            dataset / "rotary_switch" / "images",
            None,
            None,
            rows,
            ops,
        )
        validate_ops(ops)
        apply_ops(ops)
        with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        total_rows += len(rows)
        print(f"RENAMED {dataset.name}: {len(rows)} images")
    else:
        print(f"SKIP {dataset.name}: rename_manifest.csv already exists")

    dataset = ROOT / "ds05_switch_state_cls_3cls"
    manifest = dataset / "rename_manifest.csv"
    if not manifest.exists():
        rows = []
        ops = []
        for split in SPLITS:
            split_dir = dataset / split
            if not split_dir.is_dir():
                continue
            for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                group = f"{split}_{re.sub(r'[^a-z0-9]+', '_', class_dir.name.lower()).strip('_')}"
                add_group(
                    dataset,
                    "ds05",
                    group,
                    class_dir,
                    None,
                    dataset / "gt_visualization" / split / class_dir.name,
                    rows,
                    ops,
                )
        validate_ops(ops)
        apply_ops(ops)
        with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        total_rows += len(rows)
        print(f"RENAMED {dataset.name}: {len(rows)} images and GT previews")
    else:
        print(f"SKIP {dataset.name}: rename_manifest.csv already exists")

    print(f"DONE: {total_rows} source images renamed in this run")


if __name__ == "__main__":
    main()
