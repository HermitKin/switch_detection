from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIRECTIONS = ("上", "右上", "右", "右下", "下", "左下", "左", "左上")


def font(size: int):
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def parse_labels(path: Path, width: int, height: int) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 9:
            raise ValueError(f"{path}:{line_number}: expected OBB row with 9 values")
        class_id = int(values[0])
        coords = np.array([float(value) for value in values[1:]], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        result.setdefault(class_id, []).append(coords)
    return result


def obb_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    center = points.mean(axis=0)
    edge01 = points[1] - points[0]
    edge12 = points[2] - points[1]
    if np.linalg.norm(edge01) >= np.linalg.norm(edge12):
        unit = edge01 / np.linalg.norm(edge01)
        length = float(np.linalg.norm(edge01))
        width = float(np.linalg.norm(edge12))
    else:
        unit = edge12 / np.linalg.norm(edge12)
        length = float(np.linalg.norm(edge12))
        width = float(np.linalg.norm(edge01))
    end0 = center - unit * length / 2
    end1 = center + unit * length / 2
    return center, end0, end1, length, width


def pair_by_center(switches: list[np.ndarray], angles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    available = set(range(len(switches)))
    pairs = []
    for angle in sorted(angles, key=lambda points: (float(points[:, 1].mean()), float(points[:, 0].mean()))):
        angle_center = angle.mean(axis=0)
        index = min(available, key=lambda idx: float(np.linalg.norm(switches[idx].mean(axis=0) - angle_center)))
        available.remove(index)
        pairs.append((switches[index], angle))
    return pairs


def sample_end_darkness(gray: np.ndarray, center: np.ndarray, unit: np.ndarray, length: float, width: float) -> tuple[float, float]:
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    # Sample the inner 15%-40% region from each end. The base usually overlaps the dark circular hub.
    ts0 = np.linspace(-0.35 * length, -0.10 * length, 28)
    ts1 = np.linspace(0.10 * length, 0.35 * length, 28)
    ss = np.linspace(-0.42 * width, 0.42 * width, 24)

    def region_values(ts: np.ndarray) -> np.ndarray:
        grid_t, grid_s = np.meshgrid(ts, ss)
        xs = center[0] + grid_t * unit[0] + grid_s * normal[0]
        ys = center[1] + grid_t * unit[1] + grid_s * normal[1]
        return cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    region0 = region_values(ts0)
    region1 = region_values(ts1)
    combined = np.concatenate([region0.ravel(), region1.ravel()]).astype(np.uint8)
    threshold, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark0 = float(np.mean(region0 < threshold))
    dark1 = float(np.mean(region1 < threshold))
    return dark0, dark1


def direction_from_vector(vector: np.ndarray) -> tuple[float, str]:
    # Clockwise angle: 0°=up, 90°=right, 180°=down, 270°=left.
    degrees = (math.degrees(math.atan2(float(vector[0]), float(-vector[1]))) + 360.0) % 360.0
    index = int((degrees + 22.5) // 45) % 8
    return degrees, DIRECTIONS[index]


def draw_arrow(draw: ImageDraw.ImageDraw, start: np.ndarray, end: np.ndarray, color, width: int) -> None:
    start_xy = tuple(map(float, start))
    end_xy = tuple(map(float, end))
    draw.line((start_xy, end_xy), fill=color, width=width)
    vector = end - start
    vector = vector / max(float(np.linalg.norm(vector)), 1.0)
    normal = np.array([-vector[1], vector[0]])
    head_length = width * 4.5
    head_width = width * 2.6
    base = end - vector * head_length
    p1 = base + normal * head_width
    p2 = base - normal * head_width
    draw.polygon([end_xy, tuple(map(float, p1)), tuple(map(float, p2))], fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate switch-handle direction from paired OBB labels.")
    parser.add_argument("dataset", type=Path, help="Dataset containing images/train and labels/train")
    parser.add_argument("--output", type=Path, help="Output directory (default: <dataset>/orientation_demo)")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve() if args.output else dataset / "orientation_demo"
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")

    if output.exists():
        shutil.rmtree(output)
    overlays = output / "overlays"
    overlays.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    overlay_paths: list[Path] = []

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        labels = parse_labels(label_dir / f"{image_path.stem}.txt", width, height)
        if len(labels[0]) != len(labels[1]):
            raise RuntimeError(f"Unequal switch/angle counts in {image_path.name}")
        pairs = pair_by_center(labels[0], labels[1])
        draw = ImageDraw.Draw(image, "RGBA")
        label_font = font(max(18, round(min(width, height) / 32)))
        line_width = max(4, round(min(width, height) / 180))

        for object_index, (switch_points, angle_points) in enumerate(pairs, start=1):
            switch_center = switch_points.mean(axis=0)
            angle_center, end0, end1, length, angle_width = obb_geometry(angle_points)
            unit = (end1 - end0) / length
            distance0 = float(np.linalg.norm(end0 - switch_center))
            distance1 = float(np.linalg.norm(end1 - switch_center))
            geometry_tip = 0 if distance0 > distance1 else 1
            geometry_conf = abs(distance0 - distance1) / max(length, 1.0)

            dark0, dark1 = sample_end_darkness(gray, angle_center, unit, length, angle_width)
            # The handle tip is usually narrower/lighter than the end overlapping the circular hub.
            visual_tip = 0 if dark0 < dark1 else 1
            visual_conf = abs(dark0 - dark1)

            if geometry_tip == visual_tip:
                tip_index = geometry_tip
                method = "geometry+appearance"
                confidence = min(1.0, geometry_conf * 4.0 + visual_conf * 1.5)
            elif geometry_conf * 4.0 >= visual_conf * 1.5:
                tip_index = geometry_tip
                method = "geometry"
                confidence = min(1.0, geometry_conf * 4.0)
            else:
                tip_index = visual_tip
                method = "appearance"
                confidence = min(1.0, visual_conf * 1.5)

            base, tip = (end1, end0) if tip_index == 0 else (end0, end1)
            degrees, direction = direction_from_vector(tip - base)
            uncertain = confidence < 0.18
            color = (255, 190, 20, 255) if uncertain else (40, 230, 100, 255)

            draw.line([tuple(point) for point in switch_points] + [tuple(switch_points[0])], fill=(255, 70, 70, 255), width=line_width)
            draw.line([tuple(point) for point in angle_points] + [tuple(angle_points[0])], fill=(50, 170, 255, 255), width=line_width)
            draw_arrow(draw, base, tip, color, line_width + 2)
            text = f"#{object_index} {direction} {degrees:.1f}°" + (" ?" if uncertain else "")
            tx, ty = float(angle_center[0]), float(angle_center[1])
            box = draw.textbbox((tx, ty), text, font=label_font)
            draw.rectangle((box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3), fill=(0, 0, 0, 185))
            draw.text((tx, ty), text, fill=color, font=label_font)

            rows.append(
                {
                    "image": image_path.name,
                    "object_index": object_index,
                    "direction": direction,
                    "clockwise_degrees_from_up": f"{degrees:.2f}",
                    "confidence_heuristic": f"{confidence:.3f}",
                    "uncertain": str(uncertain).lower(),
                    "decision_method": method,
                    "geometry_confidence": f"{geometry_conf:.4f}",
                    "end0_dark_fraction": f"{dark0:.4f}",
                    "end1_dark_fraction": f"{dark1:.4f}",
                }
            )

        destination = overlays / image_path.name
        image.save(destination, quality=93)
        overlay_paths.append(destination)

    with (output / "orientation_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    thumb_w, thumb_h = 460, 330
    columns = 3
    rows_count = math.ceil(len(overlay_paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * thumb_h), (235, 235, 235))
    for index, path in enumerate(overlay_paths):
        with Image.open(path) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
        x = index % columns * thumb_w + (thumb_w - thumb.width) // 2
        y = index // columns * thumb_h + (thumb_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    sheet_path = output / "orientation_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    summary = {
        "images": len(images),
        "switches": len(rows),
        "uncertain": sum(row["uncertain"] == "true" for row in rows),
        "angle_convention": "clockwise degrees from up",
        "direction_bins": list(DIRECTIONS),
        "important_limit": "OBB has 180-degree symmetry; arrow direction here is a heuristic inferred from paired box geometry and image appearance.",
        "results_csv": str(output / "orientation_results.csv"),
        "contact_sheet": str(sheet_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
