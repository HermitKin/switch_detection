from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .algorithm import DEFAULT_MODEL, ROOT, IMAGE_SUFFIXES, RESULT_FIELDS, ReplayDetector, SwitchDetector, load_rgb, render_result


def collect_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No supported images found: {path}")
        return images
    raise FileNotFoundError(path)


def run_headless(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output = args.output.resolve() if args.output else input_path.parent / f"{input_path.stem}_orientation_results"
    images = collect_images(input_path)
    if input_path.is_dir() and (output == input_path or input_path in output.parents):
        raise ValueError("输出目录必须位于输入图片目录之外，以避免结果再次成为输入")
    output.mkdir(parents=True, exist_ok=True)
    detector = ReplayDetector(args.replay) if args.replay else SwitchDetector()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for image_path in images:
        image = load_rgb(image_path)
        detections, elapsed_ms = detector.detect(
            image_path=image_path,
            model_path=args.model,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=args.device,
        )
        relative = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
        rendered, image_rows, summary = render_result(
            image, relative.as_posix(), detections, show_direction=args.show_direction
        )
        destination = output / relative.parent / f"{relative.name}_result.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination, quality=94)
        summary.update({"image": str(image_path), "result_image": str(destination), "inference_ms": round(elapsed_ms, 2) if detector.mode == "inference" else None, "mode": detector.mode})
        rows.extend(image_rows)
        summaries.append(summary)
        timing = f"{elapsed_ms:.1f} ms" if detector.mode == "inference" else "replay (no inference)"
        print(f"{image_path.name}: switches={summary['switches']} matched={summary['matched']} {timing}")
    with (output / "orientation_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(
            {"model": str(detector.model_path), "mode": detector.mode, "images": summaries, "objects": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="旋钮开关 OBB 检测和方向识别 GUI")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="best.pt、weights 文件夹或训练结果文件夹")
    parser.add_argument("--replay", type=Path, nargs="?", const=ROOT / "samples" / "predictions.json",
                        help="回放已保存的示例预测，不运行模型；可提供预测 JSON 路径")
    parser.add_argument("--input", type=Path, help="启动时打开的图片/文件夹；与 --no-gui 同用时进行命令行检测")
    parser.add_argument("--output", type=Path, help="命令行模式输出目录")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-direction", action="store_true", help="启用端点正反判断并显示最终方向")
    parser.add_argument("--no-gui", action="store_true", help="对 --input 批量检测后退出")
    args = parser.parse_args()
    if args.replay and args.input is None:
        args.input = ROOT / "samples"
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1 or args.imgsz <= 0:
        parser.error("conf/iou 必须在 (0,1] 内，imgsz 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    if args.no_gui:
        if args.input is None:
            raise SystemExit("--no-gui 必须同时提供 --input")
        return run_headless(args)
    from .ui import OrientationDemoApp
    app = OrientationDemoApp(args.input.resolve() if args.input else None)
    if args.replay:
        app.detector = ReplayDetector(args.replay)
        app.root.title(f"旋钮开关方向 Demo | 示例回放 | 指示端方向 v2")
    app.model_var.set(str(args.replay) if args.replay else str(args.model.resolve()))
    app.conf_var.set(args.conf)
    app.iou_var.set(args.iou)
    app.imgsz_var.set(args.imgsz)
    app.device_var.set(args.device)
    app.direction_var.set(args.show_direction)
    app._update_direction_columns()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
