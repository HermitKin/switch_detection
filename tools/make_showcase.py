"""Capture real detections and build reproducible README figures/GIFs.

Run from repository root: python -m tools.make_showcase [--capture --model WEIGHT]
Without --capture, no model or torch is needed: figures replay the saved predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from switch_orientation_demo.algorithm import (
    ROOT, ReplayDetector, SwitchDetector, get_font, load_rgb, render_result,
)

MEDIA = ROOT / "assets" / "showcase"
SAMPLES = ROOT / "samples"
BG = "#0e1726"
FG = "#e9f0fa"
MUTED = "#a7b9d0"
ACCENT = "#48dcca"


def capture(model: Path) -> None:
    import ultralytics
    if (Path(ultralytics.__file__).parent / "nn" / "extra_modules").exists():
        raise RuntimeError("Capture requires the official installed Ultralytics package, not the local fork")
    detector = SwitchDetector()
    records = []
    for path in sorted(SAMPLES.glob("*.jpg")):
        detections, elapsed = detector.detect(path, model, 0.25, 0.55, 640, "cpu")
        records.append({"image": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "original_inference_ms": round(elapsed, 2),
                        "detections": [{"class_id": d.class_id, "class_name": d.class_name,
                                        "confidence": d.confidence, "points": d.points.tolist(),
                                        "obb_angle_radians": d.obb_angle_radians} for d in detections]})
    payload = {"schema_version": 1, "provenance": {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "model": model.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "ultralytics_version": metadata.version("ultralytics"),
        "settings": {"conf": 0.25, "iou": 0.55, "imgsz": 640, "device": "cpu"},
        "description": "Actual baseline model predictions on three demonstration images; not ground truth or an accuracy benchmark."},
        "images": records}
    (SAMPLES / "predictions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def panel(title: str, subtitle: str, size=(1200, 760)) -> Image.Image:
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    d.text((32, 24), title, fill=FG, font=get_font(31))
    d.text((32, 73), subtitle, fill=MUTED, font=get_font(17))
    return im


def paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    image = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (w-image.width)//2, y + (h-image.height)//2))


def arrow(d: ImageDraw.ImageDraw, a, b, color=ACCENT, width=4):
    d.line((a, b), fill=color, width=width)
    v = np.asarray(b, float)-a
    v /= max(np.linalg.norm(v), 1)
    n = np.array([-v[1], v[0]])
    p = np.asarray(b)-v*13
    d.polygon([tuple(b), tuple(p+n*7), tuple(p-n*7)], fill=color)


def diagrams() -> None:
    im = panel("识别流程 / Pipeline", "基线 OBB 检测 + 项目自有匹配、方向估计与结果展示", (1200, 460))
    d = ImageDraw.Draw(im)
    stages = [("01 图片输入", "EXIF 校正 / RGB"), ("02 OBB 检测", "主体 + angle"),
              ("03 一对一匹配", "中心距离 / 内含奖励"), ("04 指示端估计", "几何 + 明暗 + 翻转"),
              ("05 结果输出", "箭头 / CSV / JSON")]
    for i, (title, text) in enumerate(stages):
        x = 30+i*234
        d.rounded_rectangle((x, 145, x+204, 275), radius=16, fill="#19283e", outline="#2b435f", width=2)
        d.text((x+13, 166), title, fill=ACCENT, font=get_font(21))
        d.text((x+13, 215), text, fill=FG, font=get_font(15))
        if i < 4: arrow(d, (x+208, 210), (x+229, 210))
    d.text((32, 325), "回放模式：使用真实模型保存的四角点与置信度，重新执行 03 → 05；不运行模型。", fill=MUTED, font=get_font(19))
    d.text((32, 365), "方向评分是启发式指标；样例可视化不等于独立测试集精度。", fill=MUTED, font=get_font(19))
    im.save(MEDIA / "pipeline.png")

    im = panel("方向定义 / Indicator convention", "OBB 长轴没有正反方向；指示端需要额外判定。示意图，不是检测结果。", (1200, 600))
    d = ImageDraw.Draw(im)
    center = (255, 325)
    d.ellipse((125,195,385,455), outline="#2b435f", width=3)
    for k, label in enumerate(("上 0°", "右上 45°", "右 90°", "右下 135°", "下 180°", "左下 225°", "左 270°", "左上 315°")):
        a = math.radians(k*45)
        v = np.array([math.sin(a), -math.cos(a)])
        arrow(d, center, tuple(np.asarray(center)+v*104), MUTED, 2)
        p = np.asarray(center)+v*153
        d.text(tuple(p), label, anchor="mm", fill=FG, font=get_font(17))
    d.rounded_rectangle((545,155,1150,480), radius=18, fill="#19283e")
    d.text((575,178), "两端候选 E0 / E1", fill=ACCENT, font=get_font(24))
    d.ellipse((615,270,715,370), fill="#354158", outline=MUTED, width=3)
    d.rounded_rectangle((665,299,1045,343), radius=12, fill="#080c14", outline="#28d2eb", width=3)
    d.rectangle((673,304,708,338), fill="#eeeeec")
    arrow(d, (990,321), (683,321), "#28eb6e", 5)
    d.text((575,400), "黑色伸出端 ← 启发式选择    指示端 ← 反转 180°", fill=FG, font=get_font(18))
    d.text((575,444), "v2 假设：白色指示块位于黑色长手柄的相反端", fill=MUTED, font=get_font(17))
    d.text((32,535), "图像坐标：x 向右，y 向下。θ = atan2(vx, −vy)，正上方 0°，顺时针递增。", fill=MUTED, font=get_font(20))
    im.save(MEDIA / "direction_convention.png")


def build() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    detector = ReplayDetector(SAMPLES / "predictions.json")
    frames, summaries = [], []
    gallery = panel("Switch Detection / 真实样例结果", "红框：主体包围框   青框：angle OBB   绿箭头：可信指示方向   黄箭头：需复核", (1200, 660))
    for index, path in enumerate(sorted(SAMPLES.glob("*.jpg"))):
        original = load_rgb(path)
        detections, _ = detector.detect(path, Path("unused"), 0.25, 0.55, 640, "cpu")
        rendered, rows, summary = render_result(original, path.name, detections, True)
        summary.update({"image": path.name, "mode": "replay", "objects": rows})
        summaries.append(summary)
        rendered.save(MEDIA / f"sample_{index+1}_result.jpg", quality=93)
        paste_fit(gallery, rendered, (25+index*397, 125, 370, 435))
        ImageDraw.Draw(gallery).text((30+index*397, 575), f"样例 {index+1} · {summary['switches']} 个主体 · {summary['matched']} 个匹配", fill=FG, font=get_font(17))
        frame = panel(f"示例 {index+1} / {len(detector.records)} · 预测结果回放", "静态图片序列：原图 → OBB 叠图 → 指示端方向。不是视频跟踪或实时速度展示。")
        for stage in range(3):
            current = frame.copy()
            d = ImageDraw.Draw(current)
            output, _, _ = render_result(original, path.name, detections, show_direction=stage==2)
            paste_fit(current, original, (30,145,550,500))
            paste_fit(current, original if stage==0 else output, (620,145,550,500))
            d.text((35,113), "INPUT / 原始图片", fill=MUTED, font=get_font(19))
            d.text((625,113), ("01 输入", "02 模型预测 OBB", "03 几何 + 外观方向估计")[stage], fill=ACCENT, font=get_font(19))
            d.text((32,690), f"{summary['switches']} 个主体 | {summary['matched']} 个匹配 | {summary['uncertain']} 个方向需复核 | Work in progress", fill=MUTED, font=get_font(18))
            frames.append(current)
    gallery.save(MEDIA / "result_gallery.jpg", quality=93)
    frames[0].save(MEDIA / "orientation_demo.gif", save_all=True, append_images=frames[1:], duration=1100, loop=0, optimize=True)
    (MEDIA / "results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    diagrams()
    gui_paths = [MEDIA / name for name in ("gui_overview.png", "gui_sample_2.png", "gui_sample_3.png")]
    if all(path.is_file() for path in gui_paths):
        gui_frames = []
        for path in gui_paths:
            with Image.open(path) as source:
                gui_frames.append(ImageOps.contain(source.convert("RGB"), (1200, 800), Image.Resampling.LANCZOS))
        gui_frames[0].save(MEDIA / "gui_walkthrough.gif", save_all=True, append_images=gui_frames[1:],
                           duration=1800, loop=0, optimize=True)
    print(json.dumps([{k:v for k,v in s.items() if k!='objects'} for s in summaries], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.capture:
        if args.model is None: parser.error("--capture requires --model")
        capture(args.model.resolve())
    build()


if __name__ == "__main__":
    main()
