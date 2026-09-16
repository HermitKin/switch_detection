from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, ImageStat


ROOT = Path(r"C:\Users\Lenovo\Downloads\旋钮数据集")
DS01 = ROOT / "ds01_switch_state_det_4cls"
DS04 = ROOT / "ds04_switch_state_det_3cls"
OUTPUT = ROOT / "_comparison_ds01_ds04_same_source"


def font(size: int):
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def source_key(old_relative: str) -> str:
    stem = Path(old_relative).stem
    lower = stem.lower()
    for marker in ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_bmp.rf.", "_webp.rf."):
        position = lower.rfind(marker)
        if position >= 0:
            return lower[:position]
    return lower


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as source:
        gray = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def resized_mae(left_path: Path, right_path: Path) -> float:
    with Image.open(left_path) as left_source, Image.open(right_path) as right_source:
        left = left_source.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        right = right_source.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    means = ImageStat.Stat(ImageChops.difference(left, right)).mean
    return sum(means) / len(means)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def comparison_image(ds04_path: Path, ds01_path: Path, lines: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(27)
    info_font = font(20)
    label_font = font(23)
    draw.text((25, 18), "同源名称图片对比：左侧 ds04 / 右侧 ds01", fill="black", font=title_font)
    draw.text((25, 56), "\n".join(lines), fill=(45, 45, 45), font=info_font, spacing=5)

    left = ImageOps.contain(open_rgb(ds04_path), (650, 545), Image.Resampling.LANCZOS)
    right = ImageOps.contain(open_rgb(ds01_path), (650, 545), Image.Resampling.LANCZOS)
    lx, ly = 25 + (650 - left.width) // 2, 185 + (545 - left.height) // 2
    rx, ry = 725 + (650 - right.width) // 2, 185 + (545 - right.height) // 2
    canvas.paste(left, (lx, ly))
    canvas.paste(right, (rx, ry))
    draw.rectangle((24, 184, 675, 730), outline=(230, 80, 60), width=3)
    draw.rectangle((724, 184, 1375, 730), outline=(40, 120, 220), width=3)
    draw.text((35, 695), "ds04", fill=(230, 80, 60), font=label_font)
    draw.text((735, 695), "ds01（同源候选中感知哈希最接近）", fill=(40, 120, 220), font=label_font)
    return canvas


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {OUTPUT}")
    pairs_dir = OUTPUT / "pairs"
    pairs_dir.mkdir(parents=True)

    ds01_rows = list(csv.DictReader((DS01 / "rename_manifest.csv").open(encoding="utf-8-sig")))
    ds04_rows = list(csv.DictReader((DS04 / "rename_manifest.csv").open(encoding="utf-8-sig")))
    ds01_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ds01_rows:
        ds01_by_key[source_key(row["old_image"])].append(row)

    report_rows = []
    pair_paths: list[Path] = []
    for index, row04 in enumerate(sorted(ds04_rows, key=lambda r: source_key(r["old_image"])), start=1):
        key = source_key(row04["old_image"])
        candidates = ds01_by_key.get(key, [])
        if not candidates:
            raise RuntimeError(f"No ds01 source-name match for {row04['old_image']}")
        image04 = DS04 / row04["new_image"]
        hash04 = dhash(image04)
        ranked = []
        for row01 in candidates:
            image01 = DS01 / row01["new_image"]
            ranked.append(((hash04 ^ dhash(image01)).bit_count(), str(image01).casefold(), row01, image01))
        distance, _, row01, image01 = min(ranked)
        mae = resized_mae(image04, image01)
        sha_equal = file_sha256(image04) == file_sha256(image01)

        pair_path = pairs_dir / f"pair_{index:03d}_{key[:45]}.jpg"
        visual = comparison_image(
            image04,
            image01,
            [
                f"来源标识：{key}",
                f"dHash 距离：{distance}/64    缩放后平均像素差：{mae:.2f}/255    SHA-256 完全一致：{sha_equal}",
                f"ds04: {image04.name}    ds01: {image01.name}",
            ],
        )
        visual.save(pair_path, quality=91)
        pair_paths.append(pair_path)
        report_rows.append(
            {
                "pair_id": f"pair_{index:03d}",
                "source_key": key,
                "ds04_old_image": row04["old_image"],
                "ds04_current_image": str(image04),
                "ds01_candidate_count": len(candidates),
                "ds01_old_image_closest": row01["old_image"],
                "ds01_current_image_closest": str(image01),
                "dhash_distance_0_to_64": distance,
                "resized_rgb_mae_0_to_255": f"{mae:.4f}",
                "sha256_equal": str(sha_equal).lower(),
                "comparison_image": str(pair_path),
            }
        )

    report = OUTPUT / "ds01_ds04_302_pairs.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    # Twelve evenly spaced examples, arranged as a 3 x 4 overview.
    selected_indices = [round(i * (len(pair_paths) - 1) / 11) for i in range(12)]
    sheet = Image.new("RGB", (1440, 1120), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "ds04 与 ds01 同源名称样本抽查（每格左 ds04 / 右 ds01）", fill="black", font=font(27))
    for slot, pair_index in enumerate(selected_indices):
        pair = open_rgb(pair_paths[pair_index])
        pair.thumbnail((460, 250), Image.Resampling.LANCZOS)
        x = 15 + (slot % 3) * 475
        y = 70 + (slot // 3) * 260
        sheet.paste(pair, (x, y))
    sheet_path = OUTPUT / "contact_sheet_12_pairs.jpg"
    sheet.save(sheet_path, quality=92)

    distances = [int(row["dhash_distance_0_to_64"]) for row in report_rows]
    maes = [float(row["resized_rgb_mae_0_to_255"]) for row in report_rows]
    summary = {
        "matched_source_name_groups": len(report_rows),
        "sha256_equal_pairs": sum(row["sha256_equal"] == "true" for row in report_rows),
        "dhash_distance_zero_pairs": sum(value == 0 for value in distances),
        "dhash_distance_min": min(distances),
        "dhash_distance_max": max(distances),
        "dhash_distance_mean": sum(distances) / len(distances),
        "resized_rgb_mae_mean": sum(maes) / len(maes),
        "report": str(report),
        "contact_sheet": str(sheet_path),
        "pair_images": str(pairs_dir),
    }
    (OUTPUT / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()), encoding="utf-8"
    )
    print("\n".join(f"{key}: {value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
