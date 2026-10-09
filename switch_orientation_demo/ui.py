from __future__ import annotations

import csv
import json
import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageTk

from .algorithm import (
    ALGORITHM_VERSION,
    DEFAULT_MODEL,
    IMAGE_SUFFIXES,
    RESULT_FIELDS,
    SwitchDetector,
    load_rgb,
    render_result,
    save_current_result,
)

class OrientationDemoApp:
    def __init__(self, initial_input: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title(f"旋钮开关检测与方向识别 - {ALGORITHM_VERSION}")
        self.root.geometry("1480x860")
        self.root.minsize(1050, 650)

        self.detector = SwitchDetector()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.images: list[Path] = []
        self.image_index = -1
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.result_rows: list[dict[str, Any]] = []
        self.result_summary: dict[str, Any] = {}
        self.busy = False

        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.55)
        self.imgsz_var = tk.IntVar(value=640)
        self.device_var = tk.StringVar(value="cpu")
        self.auto_var = tk.BooleanVar(value=True)
        self.direction_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="请选择图片或文件夹")
        self.position_var = tk.StringVar(value="0 / 0")

        self._build_ui()
        self._bind_keys()
        self.root.after(100, self._poll_events)
        if initial_input is not None:
            self.root.after(150, lambda: self.open_input(initial_input))

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="打开图片", command=self.choose_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="打开文件夹", command=self.choose_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="上一张", command=self.previous_image).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(toolbar, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, textvariable=self.position_var, width=12, anchor=tk.CENTER).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="检测当前", command=self.detect_current).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(toolbar, text="批量检测", command=self.batch_detect).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="导出当前", command=self.export_current).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="切图后自动检测", variable=self.auto_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(
            toolbar,
            text="启用方向判断",
            variable=self.direction_var,
            command=self._update_direction_columns,
        ).pack(side=tk.LEFT, padx=4)

        settings = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        settings.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(settings, text="模型").pack(side=tk.LEFT)
        ttk.Entry(settings, textvariable=self.model_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(settings, text="浏览", command=self.choose_model).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(settings, text="置信度").pack(side=tk.LEFT)
        ttk.Spinbox(settings, textvariable=self.conf_var, from_=0.01, to=0.99, increment=0.05, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Label(settings, text="IoU").pack(side=tk.LEFT)
        ttk.Spinbox(settings, textvariable=self.iou_var, from_=0.1, to=0.95, increment=0.05, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Label(settings, text="尺寸").pack(side=tk.LEFT)
        ttk.Combobox(settings, textvariable=self.imgsz_var, values=(640, 800, 960, 1280), width=6, state="readonly").pack(side=tk.LEFT, padx=3)
        ttk.Label(settings, text="设备").pack(side=tk.LEFT)
        ttk.Combobox(settings, textvariable=self.device_var, values=("0", "1", "cpu"), width=6).pack(side=tk.LEFT, padx=3)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6)
        image_frame = ttk.Frame(paned)
        result_frame = ttk.Frame(paned, width=720)
        paned.add(image_frame, weight=3)
        paned.add(result_frame, weight=3)

        self.canvas = tk.Canvas(image_frame, background="#202124", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw_image())

        columns = ("id", "switch", "angle", "obb", "axis", "direction", "degree", "dir_conf", "method")
        table_frame = ttk.Frame(result_frame)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        headings = {
            "id": "编号",
            "switch": "整体框",
            "angle": "角度框",
            "obb": "OBB原角°",
            "axis": "轴线°",
            "direction": "方向",
            "degree": "方向°",
            "dir_conf": "方向可信度",
            "method": "判定",
        }
        widths = {
            "id": 42,
            "switch": 65,
            "angle": 65,
            "obb": 72,
            "axis": 62,
            "direction": 76,
            "degree": 62,
            "dir_conf": 82,
            "method": 78,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor=tk.CENTER, stretch=column in {"direction", "method"})
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        horizontal_scroll = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        horizontal_scroll.pack(fill=tk.X)
        self.tree.configure(yscrollcommand=tree_scroll.set, xscrollcommand=horizontal_scroll.set)
        self._update_direction_columns()

        ttk.Label(result_frame, text="运行信息").pack(anchor=tk.W, pady=(8, 2))
        self.log = tk.Text(result_frame, height=9, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.X)

        status = ttk.Frame(self.root, padding=6)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<space>", lambda _event: self.detect_current())
        self.root.bind("<Control-o>", lambda _event: self.choose_image())
        self.root.bind("<Control-s>", lambda _event: self.export_current())

    def choose_model(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(title="选择 OBB 模型", filetypes=(("PyTorch 模型", "*.pt"), ("所有文件", "*.*")))
        if selected:
            self.model_var.set(selected)

    def choose_image(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="选择待检测图片",
            filetypes=(("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("所有文件", "*.*")),
        )
        if selected:
            self.open_input(Path(selected))

    def choose_folder(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="选择待检测图片文件夹")
        if selected:
            self.open_input(Path(selected))

    def open_input(self, path: Path) -> None:
        if path.is_file():
            self.images = [path.resolve()]
        elif path.is_dir():
            self.images = sorted(
                (item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda item: str(item).casefold(),
            )
        else:
            self._show_error(f"路径不存在：{path}")
            return
        if not self.images:
            self._show_error("所选路径中没有支持的图片")
            return
        self.image_index = 0
        self._load_current_image()

    def _load_current_image(self) -> None:
        if not self.images:
            return
        try:
            self.original_image = load_rgb(self.images[self.image_index])
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.display_image = self.original_image
        self.result_rows = []
        self.result_summary = {}
        self.position_var.set(f"{self.image_index + 1} / {len(self.images)}")
        self.status_var.set(str(self.images[self.image_index]))
        self._fill_table([])
        self._redraw_image()
        if self.auto_var.get():
            self.detect_current()

    def previous_image(self) -> None:
        if self.images and not self.busy:
            self.image_index = (self.image_index - 1) % len(self.images)
            self._load_current_image()

    def next_image(self) -> None:
        if self.images and not self.busy:
            self.image_index = (self.image_index + 1) % len(self.images)
            self._load_current_image()

    def _settings(self) -> dict[str, Any]:
        return {
            "model_path": Path(self.model_var.get()),
            "confidence": float(self.conf_var.get()),
            "iou": float(self.iou_var.get()),
            "image_size": int(self.imgsz_var.get()),
            "device": self.device_var.get().strip() or "cpu",
        }

    def detect_current(self) -> None:
        if self.busy:
            return
        if self.image_index < 0 or self.original_image is None:
            self._show_error("请先选择图片")
            return
        image_path = self.images[self.image_index]
        image = self.original_image.copy()
        settings = self._settings()
        show_direction = bool(self.direction_var.get())
        self._set_busy(True, f"正在检测 {image_path.name} …")
        threading.Thread(
            target=self._detect_worker,
            args=(image_path, image, settings, show_direction),
            daemon=True,
        ).start()

    def _detect_worker(
        self, image_path: Path, image: Image.Image, settings: dict[str, Any], show_direction: bool
    ) -> None:
        try:
            detections, elapsed_ms = self.detector.detect(image_path=image_path, **settings)
            rendered, rows, summary = render_result(
                image, image_path.name, detections, show_direction=show_direction
            )
            summary.update({"image": str(image_path), "inference_ms": round(elapsed_ms, 2), "model": str(self.detector.model_path)})
            summary["mode"] = self.detector.mode
            if self.detector.mode == "replay": summary["inference_ms"] = None
            self.events.put(("current_done", (rendered, rows, summary)))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def batch_detect(self) -> None:
        if self.busy:
            return
        if not self.images:
            self._show_error("请先选择图片或文件夹")
            return
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="选择批量结果输出文件夹")
        if not selected:
            return
        settings = self._settings()
        show_direction = bool(self.direction_var.get())
        self._set_busy(True, f"开始批量检测，共 {len(self.images)} 张 …")
        threading.Thread(
            target=self._batch_worker,
            args=(list(self.images), Path(selected), settings, show_direction),
            daemon=True,
        ).start()

    def _batch_worker(
        self, images: list[Path], output: Path, settings: dict[str, Any], show_direction: bool
    ) -> None:
        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        try:
            common_root = Path(os.path.commonpath([str(path.parent) for path in images])).resolve()
            if output.resolve() == common_root or common_root in output.resolve().parents:
                raise ValueError("输出目录必须位于输入图片目录之外")
            output.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(images, 1):
                image = load_rgb(image_path)
                detections, elapsed_ms = self.detector.detect(image_path=image_path, **settings)
                relative = image_path.relative_to(common_root)
                rendered, rows, summary = render_result(
                    image, relative.as_posix(), detections, show_direction=show_direction
                )
                destination = output / relative.parent / f"{relative.name}_result.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(destination, quality=94)
                summary.update({"image": str(image_path), "result_image": str(destination), "inference_ms": round(elapsed_ms, 2), "mode": self.detector.mode})
                if self.detector.mode == "replay": summary["inference_ms"] = None
                all_rows.extend(rows)
                summaries.append(summary)
                self.events.put(("progress", f"批量检测 {index}/{len(images)}：{image_path.name}"))

            with (output / "orientation_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(all_rows)
            (output / "summary.json").write_text(
                json.dumps(
                    {"model": str(self.detector.model_path), "mode": self.detector.mode, "images": summaries, "objects": all_rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.events.put(("batch_done", (output, len(images), len(all_rows))))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def export_current(self) -> None:
        if self.display_image is None or not self.result_rows:
            self._show_error("当前没有可导出的检测结果")
            return
        from tkinter import filedialog

        stem = self.images[self.image_index].stem if self.image_index >= 0 else "result"
        selected = filedialog.asksaveasfilename(
            title="保存当前结果（同时生成 CSV 和 JSON）",
            defaultextension=".jpg",
            initialfile=f"{stem}_result.jpg",
            filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png")),
        )
        if selected:
            save_current_result(Path(selected), self.display_image, self.result_rows, self.result_summary)
            self.status_var.set(f"已导出：{selected}")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "current_done":
                    rendered, rows, summary = payload
                    self.display_image = rendered
                    self.result_rows = rows
                    self.result_summary = summary
                    self._fill_table(rows)
                    self._redraw_image()
                    self._write_log(summary)
                    timing = "示例回放（不运行模型）" if summary.get("mode") == "replay" else f"耗时 {summary['inference_ms']:.1f} ms"
                    self._set_busy(False, f"完成：{summary['switches']} 个开关，{timing}")
                elif event == "batch_done":
                    output, image_count, object_count = payload
                    self._set_busy(False, f"批量完成：{image_count} 张，{object_count} 个开关，结果在 {output}")
                elif event == "progress":
                    self.status_var.set(str(payload))
                elif event == "error":
                    self._set_busy(False, "运行失败")
                    self._show_error(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _fill_table(self, rows: list[dict[str, Any]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            reliable = row["direction_reliable"]
            if not row["direction"]:
                direction = "—"
            elif reliable or row["direction"] == "未检测到angle":
                direction = row["direction"]
            else:
                direction = f"{row['direction']} ?"
            self.tree.insert(
                "",
                "end",
                values=(
                    row["object_index"],
                    format_number(row["switch_confidence"]),
                    format_number(row["angle_confidence"]),
                    format_number(row["obb_angle_degrees"], 1),
                    format_number(row["axis_degrees"], 1),
                    direction,
                    format_number(row["direction_degrees"], 1),
                    format_number(row["direction_confidence"]),
                    row["decision_method"],
                ),
            )

    def _update_direction_columns(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.configure(
            displaycolumns="#all" if self.direction_var.get() else ("id", "switch", "angle", "obb")
        )

    def _write_log(self, summary: dict[str, Any]) -> None:
        lines = [
            f"运行模式：{'示例回放（不运行模型）' if summary.get('mode') == 'replay' else '模型推理'}",
            f"图片：{summary['image']}",
            f"模型：{summary['model']}",
            f"原始检测：{summary['detections']}",
            f"switch_handle：{summary['switches']}",
            f"成功匹配 angle：{summary['matched']}",
            f"未匹配 angle：{summary['unmatched_angles']}",
            "推理耗时：未执行模型" if summary.get("mode") == "replay" else f"推理耗时：{summary['inference_ms']:.2f} ms",
        ]
        if summary["direction_enabled"]:
            lines.extend(
                (
                    f"方向低可信：{summary['uncertain']}",
                    f"方向算法：{ALGORITHM_VERSION}",
                    "方向判断：已启用（包含180°端点启发式判断）",
                )
            )
        else:
            lines.append("方向判断：未启用，仅显示模型原始OBB角度")
        self.log.configure(state=self.tk.NORMAL)
        self.log.delete("1.0", self.tk.END)
        self.log.insert(self.tk.END, "\n".join(lines))
        self.log.configure(state=self.tk.DISABLED)

    def _redraw_image(self) -> None:
        if self.display_image is None:
            self.canvas.delete("all")
            return
        width = max(self.canvas.winfo_width() - 16, 100)
        height = max(self.canvas.winfo_height() - 16, 100)
        preview = ImageOps.contain(self.display_image, (width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo, anchor=self.tk.CENTER)

    def _set_busy(self, busy: bool, message: str) -> None:
        self.busy = busy
        self.status_var.set(message)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _show_error(self, message: str) -> None:
        from tkinter import messagebox

        messagebox.showerror("错误", message)

    def run(self) -> None:
        self.root.mainloop()


def format_number(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.{digits}f}"
