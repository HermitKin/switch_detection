from __future__ import annotations

import zipfile
from pathlib import Path


DATASET = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集\ds06_switch_handle_small_300")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    for source_split in SPLITS:
        images = sorted(
            path
            for path in (DATASET / source_split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted((DATASET / source_split / "labels").glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError(f"Image-label mismatch in {source_split}")

        output = DATASET.parent / f"ds06_{source_split}_cvat_ultralytics_detection.zip"
        if output.exists():
            output.unlink()
        data_yaml = "path: ./\ntrain: train.txt\nnames:\n  0: switch_handle\n"
        image_list = "".join(f"images/train/{path.name}\n" for path in images)

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("data.yaml", data_yaml)
            archive.writestr("train.txt", image_list)
            for image in images:
                archive.write(image, f"images/train/{image.name}")
            for label in labels:
                archive.write(label, f"labels/train/{label.name}")

        with zipfile.ZipFile(output, "r") as archive:
            expected = 2 + len(images) + len(labels)
            if len(archive.namelist()) != expected:
                raise RuntimeError(f"Unexpected entry count in {output}")
            if archive.namelist()[:2] != ["data.yaml", "train.txt"]:
                raise RuntimeError(f"Invalid ZIP root in {output}")
        print(f"{source_split}: images={len(images)}, labels={len(labels)}, bytes={output.stat().st_size}, file={output}")


if __name__ == "__main__":
    main()
