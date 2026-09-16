from __future__ import annotations

import zipfile
from pathlib import Path


DATASET = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集\ds06_switch_handle_small_300")
OUTPUT = DATASET.parent / "ds06_switch_handle_small_300_cvat_ultralytics_detection.zip"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()

    images_by_split: dict[str, list[Path]] = {}
    labels_by_split: dict[str, list[Path]] = {}
    for split in SPLITS:
        images = sorted(
            path
            for path in (DATASET / split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted((DATASET / split / "labels").glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise RuntimeError(f"Image-label mismatch in {split}")
        images_by_split[split] = images
        labels_by_split[split] = labels

    data_yaml = (
        "path: ./\n"
        "train: train.txt\n"
        "val: valid.txt\n"
        "test: test.txt\n"
        "names:\n"
        "  0: switch_handle\n"
    )

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("data.yaml", data_yaml)
        for split in SPLITS:
            image_list = "".join(f"images/{split}/{path.name}\n" for path in images_by_split[split])
            archive.writestr(f"{split}.txt", image_list)
            for image in images_by_split[split]:
                archive.write(image, f"images/{split}/{image.name}")
            for label in labels_by_split[split]:
                archive.write(label, f"labels/{split}/{label.name}")

    with zipfile.ZipFile(OUTPUT, "r") as archive:
        names = archive.namelist()
        if not names or names[0] != "data.yaml":
            raise RuntimeError("data.yaml is not at ZIP root")
        if len(names) != 604:  # yaml + 3 lists + 300 images + 300 labels
            raise RuntimeError(f"Unexpected ZIP entry count: {len(names)}")
        bad_labels = []
        for name in names:
            if not name.startswith("labels/") or not name.endswith(".txt"):
                continue
            for line in archive.read(name).decode("utf-8").splitlines():
                values = line.split()
                if len(values) != 5 or values[0] != "0":
                    bad_labels.append((name, line))
        if bad_labels:
            raise RuntimeError(f"Invalid labels in ZIP: {bad_labels[:3]}")

    print(f"output={OUTPUT}")
    print(f"bytes={OUTPUT.stat().st_size}")
    print("entries=604")
    print("images=300")
    print("labels=300")


if __name__ == "__main__":
    main()
