from __future__ import annotations

import csv
import json
import random
import shutil
import uuid
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集")
OUTPUT = ROOT / "ds06_switch_handle_small_300"
SEED = 20260915
SOURCES = {
    "ds01_switch_state_det_4cls": {"train": 140, "valid": 40, "test": 20},
    "ds02_rotary_switch_det_1cls": {"train": 70, "valid": 20, "test": 10},
}


def source_key(old_relative: str, dataset_name: str) -> str:
    stem = Path(old_relative).stem.lower()
    for marker in ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_bmp.rf.", "_webp.rf."):
        position = stem.rfind(marker)
        if position >= 0:
            stem = stem[:position]
            break
    # ds01 and ds02 are unrelated sources, so short original names cannot collide accidentally.
    return f"{dataset_name}:{stem}"


def validate_label(label_path: Path) -> tuple[int, str]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty label file: {label_path}")
    for line_number, line in enumerate(lines, start=1):
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            raise RuntimeError(f"Expected class-0 HBB label at {label_path}:{line_number}: {line}")
        coordinates = [float(value) for value in values[1:]]
        if any(value < 0 or value > 1 for value in coordinates):
            raise RuntimeError(f"Out-of-range coordinate at {label_path}:{line_number}")
    return len(lines), "\n".join(lines) + "\n"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {OUTPUT}")
    staging = ROOT / f".{OUTPUT.name}.building-{uuid.uuid4().hex}"
    rng = random.Random(SEED)
    manifest_rows: list[dict[str, object]] = []
    counters = {"train": 0, "valid": 0, "test": 0}
    source_counts: dict[str, dict[str, int]] = {}

    try:
        for split in counters:
            (staging / split / "images").mkdir(parents=True, exist_ok=True)
            (staging / split / "labels").mkdir(parents=True, exist_ok=True)

        for dataset_name, allocation in SOURCES.items():
            dataset = ROOT / dataset_name
            rows = list(csv.DictReader((dataset / "rename_manifest.csv").open(encoding="utf-8-sig")))
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                groups[source_key(row["old_image"], dataset_name)].append(row)

            required = sum(allocation.values())
            group_keys = sorted(groups)
            if len(group_keys) < required:
                raise RuntimeError(f"{dataset_name} has only {len(group_keys)} source groups; {required} required")
            selected_keys = rng.sample(group_keys, required)
            rng.shuffle(selected_keys)
            source_counts[dataset_name] = {}
            cursor = 0

            for target_split in ("train", "valid", "test"):
                count = allocation[target_split]
                keys_for_split = selected_keys[cursor : cursor + count]
                cursor += count
                source_counts[dataset_name][target_split] = count

                for key in keys_for_split:
                    # Keep a single deterministic export/augmentation for every original source group.
                    candidates = sorted(groups[key], key=lambda row: row["new_image"].casefold())
                    source_row = candidates[rng.randrange(len(candidates))]
                    source_image = dataset / source_row["new_image"]
                    source_label = dataset / source_row["new_label"]
                    object_count, normalized_label = validate_label(source_label)

                    counters[target_split] += 1
                    new_stem = f"ds06_{target_split}_{counters[target_split]:06d}"
                    target_image = staging / target_split / "images" / f"{new_stem}{source_image.suffix.lower()}"
                    target_label = staging / target_split / "labels" / f"{new_stem}.txt"
                    shutil.copy2(source_image, target_image)
                    target_label.write_text(normalized_label, encoding="utf-8")

                    manifest_rows.append(
                        {
                            "target_split": target_split,
                            "target_image": str(target_image.relative_to(staging)),
                            "target_label": str(target_label.relative_to(staging)),
                            "source_dataset": dataset_name,
                            "source_group": key,
                            "source_variant_count": len(candidates),
                            "source_original_image": source_row["old_image"],
                            "source_current_image": str(source_image),
                            "source_current_label": str(source_label),
                            "object_count": object_count,
                        }
                    )

        expected = {"train": 210, "valid": 60, "test": 30}
        if counters != expected:
            raise RuntimeError(f"Unexpected split counts: {counters}")
        group_keys = [str(row["source_group"]) for row in manifest_rows]
        if len(group_keys) != len(set(group_keys)):
            raise RuntimeError("Source-group leakage detected")

        with (staging / "selection_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

        (staging / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: [switch_handle]\n",
            encoding="utf-8",
        )
        summary = {
            "dataset": OUTPUT.name,
            "purpose": "small validation dataset for CVAT OBB annotation and pipeline verification",
            "seed": SEED,
            "total_images": sum(counters.values()),
            "split_counts": counters,
            "split_ratio": "7:2:1",
            "source_counts": source_counts,
            "class_names": ["switch_handle"],
            "current_label_format": "YOLO horizontal bounding box (class cx cy w h)",
            "target_label_format": "YOLO OBB after manual rotation/refinement in CVAT",
            "source_group_leakage": False,
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "README.txt").write_text(
            "小型验证数据集，共 300 张。\n"
            "train/valid/test = 210/60/30（7:2:1）。\n"
            "ds01 提供 200 张，ds02 提供 100 张。\n"
            "每个原始来源组只选一个版本，避免增强图片跨集合泄漏。\n"
            "当前标签为 switch_handle 水平框，可导入 CVAT 后旋转并收紧为 OBB。\n",
            encoding="utf-8",
        )
        staging.rename(OUTPUT)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
