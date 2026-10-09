from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DIRECTIONS = ("上", "右上", "右", "右下", "下", "左下", "左", "左上")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "switch_yolo11n_obb.pt"
ALGORITHM_VERSION = "指示端方向 v2"
RESULT_FIELDS = (
    "image",
    "object_index",
    "switch_confidence",
    "angle_confidence",
    "obb_angle_radians",
    "obb_angle_degrees",
    "axis_degrees",
    "direction",
    "direction_degrees",
    "direction_confidence",
    "direction_reliable",
    "decision_method",
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    points: np.ndarray
    obb_angle_radians: float

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)


@dataclass
class MatchedSwitch:
    switch: Detection
    angle: Detection | None


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def resolve_model_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "weights" / "best.pt",
        path / "best.pt",
        path / "weights" / "best_fp32.pt",
        path / "best_fp32.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到模型文件：{path}")


class SwitchDetector:
    def __init__(self) -> None:
        self.model: Any = None
        self.model_path: Path | None = None
        self.mode = "inference"

    def load(self, model_path: Path) -> None:
        resolved = resolve_model_path(model_path)
        if self.model is not None and self.model_path == resolved:
            return
        from ultralytics import YOLO

        model = YOLO(str(resolved))
        if getattr(model, "task", None) != "obb":
            raise RuntimeError(f"模型任务应为 OBB，实际为 {getattr(model, 'task', None)}")
        if model.names[0] != "switch_handle" or model.names[1] != "angle":
            raise ValueError("模型类别必须为 0: switch_handle、1: angle，请提供项目训练后的权重")
        self.model = model
        self.model_path = resolved

    def detect(
        self,
        image_path: Path,
        model_path: Path,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
    ) -> tuple[list[Detection], float]:
        self.load(model_path)
        started = time.perf_counter()
        result = self.model.predict(
            source=str(image_path),
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.obb is None or len(result.obb) == 0:
            return [], elapsed_ms

        corners = result.obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
        obb_angles = result.obb.xywhr[:, 4].detach().cpu().numpy().astype(float)
        classes = result.obb.cls.detach().cpu().numpy().astype(int)
        confidences = result.obb.conf.detach().cpu().numpy().astype(float)
        names = result.names
        detections: list[Detection] = []
        for points, class_id, score, obb_angle in zip(corners, classes, confidences, obb_angles):
            name = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
            detections.append(
                Detection(int(class_id), str(name), float(score), points.reshape(4, 2), float(obb_angle))
            )
        return detections, elapsed_ms


class ReplayDetector:
    """Replay saved model detections only for the exact bundled sample images."""

    def __init__(self, path: Path) -> None:
        self.model_path = path.resolve()
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported replay schema")
        self.records = {item["image"]: item for item in payload["images"]}
        self.mode = "replay"
        self.provenance = payload["provenance"]

    def detect(self, image_path: Path, model_path: Path, confidence: float,
               iou: float, image_size: int, device: str) -> tuple[list[Detection], float]:
        record = self.records.get(image_path.name)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if record is None or digest != record["sha256"]:
            raise ValueError("回放只支持 samples/ 中的原始示例图片；任意新图片请使用模型推理模式")
        settings = self.provenance["settings"]
        if confidence < settings["conf"] or iou != settings["iou"] or image_size != settings["imgsz"]:
            raise ValueError("回放保留原始 NMS 结果，请使用 conf>=0.25、iou=0.55、imgsz=640")
        detections = [Detection(int(item["class_id"]), item["class_name"],
                                float(item["confidence"]), np.asarray(item["points"], dtype=np.float32),
                                float(item["obb_angle_radians"]))
                      for item in record["detections"] if item["confidence"] >= confidence]
        return detections, 0.0


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def is_switch(det: Detection) -> bool:
    return det.class_name == "switch_handle" or det.class_id == 0


def is_angle(det: Detection) -> bool:
    return det.class_name == "angle" or det.class_id == 1


def match_switches(detections: list[Detection]) -> tuple[list[MatchedSwitch], list[Detection]]:
    switches = [det for det in detections if is_switch(det)]
    angles = [det for det in detections if is_angle(det)]
    candidates: list[tuple[float, int, int]] = []
    for switch_index, switch in enumerate(switches):
        scale = max(math.sqrt(max(polygon_area(switch.points), 1.0)), 1.0)
        for angle_index, angle in enumerate(angles):
            distance = float(np.linalg.norm(switch.center - angle.center)) / scale
            inside = cv2.pointPolygonTest(switch.points.astype(np.float32), tuple(map(float, angle.center)), False) >= 0
            score = distance - (0.7 if inside else 0.0)
            if inside or distance <= 0.9:
                candidates.append((score, switch_index, angle_index))

    used_switches: set[int] = set()
    used_angles: set[int] = set()
    pair_by_switch: dict[int, Detection] = {}
    for _, switch_index, angle_index in sorted(candidates):
        if switch_index in used_switches or angle_index in used_angles:
            continue
        used_switches.add(switch_index)
        used_angles.add(angle_index)
        pair_by_switch[switch_index] = angles[angle_index]

    ordered = sorted(enumerate(switches), key=lambda item: (float(item[1].center[1]), float(item[1].center[0])))
    matches = [MatchedSwitch(switch, pair_by_switch.get(index)) for index, switch in ordered]
    unmatched_angles = [angle for index, angle in enumerate(angles) if index not in used_angles]
    return matches, unmatched_angles


def obb_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    center = points.mean(axis=0)
    edges = [points[(index + 1) % 4] - points[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_index = int(np.argmax(lengths))
    unit = edges[long_index] / max(lengths[long_index], 1e-6)
    length = lengths[long_index]
    width = lengths[(long_index + 1) % 4]
    return center, center - unit * length / 2, center + unit * length / 2, length, width


def sample_end_darkness(
    gray: np.ndarray, center: np.ndarray, unit: np.ndarray, length: float, width: float
) -> tuple[float, float]:
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    ts0 = np.linspace(-0.40 * length, -0.10 * length, 28)
    ts1 = np.linspace(0.10 * length, 0.40 * length, 28)
    ss = np.linspace(-0.40 * width, 0.40 * width, 20)

    def values(ts: np.ndarray) -> np.ndarray:
        grid_t, grid_s = np.meshgrid(ts, ss)
        xs = center[0] + grid_t * unit[0] + grid_s * normal[0]
        ys = center[1] + grid_t * unit[1] + grid_s * normal[1]
        return cv2.remap(
            gray,
            xs.astype(np.float32),
            ys.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    region0, region1 = values(ts0), values(ts1)
    combined = np.concatenate((region0.ravel(), region1.ravel())).astype(np.uint8)
    threshold, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.mean(region0 < threshold)), float(np.mean(region1 < threshold))


def clockwise_from_up(vector: np.ndarray) -> float:
    return (math.degrees(math.atan2(float(vector[0]), float(-vector[1]))) + 360.0) % 360.0


def undirected_axis_degrees(vector: np.ndarray) -> float:
    return clockwise_from_up(vector) % 180.0


def direction_name(degrees: float) -> str:
    return DIRECTIONS[int((degrees + 22.5) // 45) % 8]


def estimate_direction(image: Image.Image, switch: Detection, angle: Detection) -> dict[str, Any]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    center, end0, end1, length, width = obb_geometry(angle.points)
    unit = (end1 - end0) / max(length, 1e-6)

    # The endpoint farther from the whole-switch center is usually the handle tip.
    distance0 = float(np.linalg.norm(end0 - switch.center))
    distance1 = float(np.linalg.norm(end1 - switch.center))
    geometry_tip = 0 if distance0 > distance1 else 1
    geometry_confidence = min(1.0, abs(distance0 - distance1) / max(length * 0.25, 1.0))

    # The end overlapping the circular hub is commonly darker; the lighter end is the tip.
    dark0, dark1 = sample_end_darkness(gray, center, unit, length, width)
    appearance_tip = 0 if dark0 < dark1 else 1
    appearance_confidence = min(1.0, abs(dark0 - dark1) / 0.30)

    if geometry_tip == appearance_tip:
        tip_index = geometry_tip
        method = "几何+外观"
        heuristic_confidence = 0.55 * geometry_confidence + 0.45 * appearance_confidence
    elif geometry_confidence >= appearance_confidence:
        tip_index = geometry_tip
        method = "几何"
        heuristic_confidence = geometry_confidence * 0.65
    else:
        tip_index = appearance_tip
        method = "外观"
        heuristic_confidence = appearance_confidence * 0.65

    # 上面的启发式选出黑色长手柄伸出的一端；这类选择开关真正的档位
    # 指向由另一端的白色指示块表示，因此将箭头反转 180 度。
    lever_base, lever_tip = (end1, end0) if tip_index == 0 else (end0, end1)
    base, tip = lever_tip, lever_base
    direction_degrees = clockwise_from_up(tip - base)
    axis_degrees = undirected_axis_degrees(end1 - end0)
    confidence = float(np.clip(heuristic_confidence * angle.confidence, 0.0, 1.0))
    return {
        "base": base,
        "tip": tip,
        "axis_degrees": axis_degrees,
        "direction_degrees": direction_degrees,
        "direction": direction_name(direction_degrees),
        "direction_confidence": confidence,
        "direction_reliable": confidence >= 0.35,
        "decision_method": f"{method}+指示端",
    }


def draw_arrow(draw: ImageDraw.ImageDraw, start: np.ndarray, end: np.ndarray, color: tuple[int, ...], width: int) -> None:
    draw.line((tuple(start), tuple(end)), fill=color, width=width)
    vector = end - start
    vector = vector / max(float(np.linalg.norm(vector)), 1.0)
    normal = np.array([-vector[1], vector[0]], dtype=np.float32)
    base = end - vector * width * 4.5
    p1 = base + normal * width * 2.2
    p2 = base - normal * width * 2.2
    draw.polygon((tuple(end), tuple(p1), tuple(p2)), fill=color)


def horizontal_box(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def render_result(
    image: Image.Image,
    image_name: str,
    detections: list[Detection],
    show_direction: bool = False,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, int]]:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    matches, unmatched_angles = match_switches(detections)
    short_side = min(canvas.size)
    line_width = max(2, round(short_side / 240))
    label_font = get_font(max(14, round(short_side / 38)))
    rows: list[dict[str, Any]] = []

    for object_index, match in enumerate(matches, 1):
        box = horizontal_box(match.switch.points)
        draw.rectangle(box, outline=(255, 55, 70, 255), width=line_width)
        switch_text = f"#{object_index} switch {match.switch.confidence:.2f}"
        draw_label(draw, (box[0], box[1]), switch_text, label_font, (255, 80, 80, 255))

        row: dict[str, Any] = {
            "image": image_name,
            "object_index": object_index,
            "switch_confidence": round(match.switch.confidence, 4),
            "angle_confidence": "",
            "obb_angle_radians": "",
            "obb_angle_degrees": "",
            "axis_degrees": "",
            "direction": "未检测到angle",
            "direction_degrees": "",
            "direction_confidence": "",
            "direction_reliable": False,
            "decision_method": "",
        }
        if match.angle is not None:
            points = match.angle.points
            draw.line([tuple(point) for point in points] + [tuple(points[0])], fill=(30, 210, 235, 255), width=line_width)
            obb_angle_degrees = math.degrees(match.angle.obb_angle_radians)
            row.update(
                {
                    "angle_confidence": round(match.angle.confidence, 4),
                    "obb_angle_radians": round(match.angle.obb_angle_radians, 6),
                    "obb_angle_degrees": round(obb_angle_degrees, 2),
                }
            )
            if show_direction:
                estimate = estimate_direction(image, match.switch, match.angle)
                reliable = bool(estimate["direction_reliable"])
                arrow_color = (40, 235, 110, 255) if reliable else (255, 185, 30, 255)
                draw_arrow(draw, estimate["base"], estimate["tip"], arrow_color, line_width + 1)
                label = (
                    f"OBB {obb_angle_degrees:.1f}°\n"
                    f"{estimate['direction']} {estimate['direction_degrees']:.1f}° "
                    f"({estimate['direction_confidence']:.2f})"
                )
                if not reliable:
                    label += " ?"
                draw_label(draw, tuple(match.angle.center), label, label_font, arrow_color)
                row.update(
                    {
                        "axis_degrees": round(float(estimate["axis_degrees"]), 2),
                        "direction": str(estimate["direction"]),
                        "direction_degrees": round(float(estimate["direction_degrees"]), 2),
                        "direction_confidence": round(float(estimate["direction_confidence"]), 4),
                        "direction_reliable": reliable,
                        "decision_method": str(estimate["decision_method"]),
                    }
                )
            else:
                row["direction"] = ""
                draw_label(
                    draw,
                    tuple(match.angle.center),
                    f"OBB {obb_angle_degrees:.1f}°",
                    label_font,
                    (30, 210, 235, 255),
                )
        rows.append(row)

    for angle in unmatched_angles:
        points = angle.points
        draw.line([tuple(point) for point in points] + [tuple(points[0])], fill=(255, 70, 230, 255), width=line_width)
        raw_degrees = math.degrees(angle.obb_angle_radians)
        draw_label(
            draw,
            tuple(angle.center),
            f"未匹配 angle {angle.confidence:.2f} | OBB {raw_degrees:.1f}°",
            label_font,
            (255, 70, 230, 255),
        )

    summary = {
        "detections": len(detections),
        "switches": len(matches),
        "matched": sum(match.angle is not None for match in matches),
        "unmatched_angles": len(unmatched_angles),
        "direction_enabled": show_direction,
        "uncertain": sum(
            row["direction"] not in {"", "未检测到angle"} and not row["direction_reliable"] for row in rows
        ),
    }
    return canvas, rows, summary


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    label_font: ImageFont.ImageFont,
    color: tuple[int, ...],
) -> None:
    x, y = position
    bounds = draw.multiline_textbbox((x, y), text, font=label_font, spacing=2)
    draw.rectangle((bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2), fill=(0, 0, 0, 185))
    draw.multiline_text((x, y), text, font=label_font, fill=color, spacing=2)


def save_current_result(destination: Path, image: Image.Image, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=94)
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(summary)
    payload["objects"] = rows
    destination.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
