from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集").resolve()
DATASET_NAME = "ds04_switch_state_det_3cls"
DATASET = (ROOT / DATASET_NAME).resolve()
BACKUP_ROOT = (ROOT / "_label_backup_before_switch_handle_20260915").resolve()
BACKUP_DATASET = (BACKUP_ROOT / DATASET_NAME).resolve()
COMPARISON = (ROOT / "_comparison_ds01_ds04_same_source").resolve()


def validate_exact_target(path: Path, expected_parent: Path, expected_name: str) -> None:
    if path.parent != expected_parent or path.name != expected_name:
        raise RuntimeError(f"Refusing unsafe target: {path}")


def main() -> None:
    validate_exact_target(DATASET, ROOT, DATASET_NAME)
    validate_exact_target(BACKUP_DATASET, BACKUP_ROOT, DATASET_NAME)
    validate_exact_target(COMPARISON, ROOT, "_comparison_ds01_ds04_same_source")

    before = {
        "dataset_files": sum(1 for path in DATASET.rglob("*") if path.is_file()) if DATASET.exists() else 0,
        "backup_files": sum(1 for path in BACKUP_DATASET.rglob("*") if path.is_file())
        if BACKUP_DATASET.exists()
        else 0,
        "comparison_files": sum(1 for path in COMPARISON.rglob("*") if path.is_file())
        if COMPARISON.exists()
        else 0,
    }

    if DATASET.exists():
        shutil.rmtree(DATASET)
    if BACKUP_DATASET.exists():
        shutil.rmtree(BACKUP_DATASET)
    if COMPARISON.exists():
        shutil.rmtree(COMPARISON)

    manifest_path = BACKUP_ROOT / "backup_manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("dataset") != DATASET_NAME]
    if not fieldnames:
        raise RuntimeError("Backup manifest has no header")
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = BACKUP_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary["backup_manifest_entries"] = len(rows)
    summary.get("backup_by_dataset", {}).pop(DATASET_NAME, None)
    summary.get("normalization", {}).pop(DATASET_NAME, None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_summary_path = ROOT / "duplicate_images_sha256_summary.json"
    if audit_summary_path.exists():
        audit = json.loads(audit_summary_path.read_text(encoding="utf-8-sig"))
        audit["scope"] = [name for name in audit.get("scope", []) if name != DATASET_NAME]
        audit["total_source_images"] = 2090
        audit["unique_image_hashes"] = 2090
        audit_summary_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "removed": before,
        "dataset_exists_after": DATASET.exists(),
        "backup_exists_after": BACKUP_DATASET.exists(),
        "comparison_exists_after": COMPARISON.exists(),
        "backup_manifest_entries_after": len(rows),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
