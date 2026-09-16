# Switch Detection

> 🚧 **开发中（Work in Progress）**：项目仍在持续整理，数据格式、脚本接口和方向判断方法可能调整。

开关与旋钮视觉识别实验代码。目前重点包括数据集审计与去重、标签规范化、OBB（旋转框）标注准备、Ground Truth 可视化，以及根据开关主体与手柄标注估计朝向的演示程序。

本仓库只保存项目自身代码，不包含数据集、训练结果、模型权重或 Ultralytics/YOLO 源码。需要的第三方包通过 `requirements.txt` 安装。

## 当前内容

- `demo_switch_orientation.py`：读取两类 OBB 标注，匹配开关主体与手柄，并输出方向角、八方向分类、可视化叠图、CSV 和 JSON 汇总。
- `prepare_obb_dataset.py`：审计多个开关数据集，检测重复与疑似近重复图像，生成去重后的 OBB 标注池和清单。
- `normalize_switch_handle_labels.py`：规范化开关手柄类别及标签格式，并保留备份。
- `visualize_gt.py`：可视化检测或分类数据集的 Ground Truth。
- `visualize_ultralytics_obb_gt.py`：可视化 Ultralytics OBB 格式标签。
- `audit_duplicate_images.py`、`compare_ds01_ds04_overlap.py`：数据集重复与来源重叠检查。
- `build_small_validation_dataset.py`、`package_ds06_*.py`：构建小型验证集并打包为 CVAT 可用格式。
- `prepare_*`、`select_pose_annotation_batch.py`：数据清洗、拆分和标注批次准备工具。

## 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

训练或推理脚本使用 PyPI 安装的 `ultralytics` 包，不需要把 YOLO 源码复制到本仓库。

## 方向演示

数据集应采用以下结构：

```text
dataset/
├─ images/train/
└─ labels/train/
```

每个标签文件使用 Ultralytics OBB 格式：类别 `0` 表示开关主体，类别 `1` 表示方向手柄；每行由类别编号和四个归一化角点组成。

```bash
python demo_switch_orientation.py path/to/dataset
```

可指定输出目录：

```bash
python demo_switch_orientation.py path/to/dataset --output path/to/output
```

程序会生成：

- `overlays/`：方向箭头与 OBB 叠图
- `orientation_results.csv`：逐目标方向结果
- `orientation_contact_sheet.jpg`：汇总预览图
- `summary.json`：运行摘要

角度定义为从正上方开始顺时针计算。由于 OBB 本身存在 180° 对称性，箭头正反方向目前结合几何关系与图像明暗启发式推断，低置信度结果会标记为不确定。

## 数据与模型

数据集、模型权重和运行输出不会提交到 Git。部分一次性数据整理脚本仍保留本地目录常量，运行前请按实际数据位置修改脚本顶部的路径。

## 状态

当前代码用于实验验证，尚未形成稳定 API 或可发布模型。
