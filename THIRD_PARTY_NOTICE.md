# 项目代码、依赖与展示素材

本仓库发布项目自身的应用层代码、数据准备工具和展示素材，模型架构由外部
Ultralytics 依赖提供。模型实验改进源码、完整训练数据和权重不纳入此版本。

- 推理依赖为官方 [`ultralytics==8.3.9`](https://github.com/ultralytics/ultralytics/tree/v8.3.9)。
  上游许可见该版本的 [LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.9/LICENSE)。
- 另使用 NumPy、OpenCV、Pillow、PyYAML 与 Python 标准库，依赖各自保留原有许可。
- 仓库目前未为项目自有代码指定单独的开源许可证；公开展示不等于额外授予任意用途许可。
- 三张 `samples/` 图片来自已有项目演示材料，原始数据来源许可待补充。
- `assets/showcase/` 中的检测叠图和 GIF 由这些样例及保存的真实模型预测生成。
  GUI 截图来自运行中的项目程序；流程图和角度图是项目原理示意。
- 示例预测没有人工方向真值，不用于宣称最终准确率、mAP 或生产部署性能。
