# ⚡ Training & Optimization | 训练与优化算法

本模块关注如何“训练好”一个模型，涵盖优化器的数学原理、显存优化技术以及加速算子的实现。

## 📝 核心内容

- **优化器 (Optimizers)**
  - 基础算法：BGD, SGD, Momentum
  - 自适应算法：**Adam** 原理 vs **AdamW** (权重衰减的区别)
- **Flash Attention 系列**
  - 核心思想：Tiling (分块) + Recomputation (重计算)
  - IO-Awareness：如何降低 HBM (高带宽显存) 的访问次数
  - 对 MQA (Multi-Query) 和 GQA (Group-Query) 的支持与处理
- **量化技术 (Quantization)**
  - 精度标准：FP32, FP16, BF16, INT8, INT4
  - 量化原理：FP16 到 INT4 的映射过程
  - 仿射量化 (Affine Quantization) 通用方法

## 🎯 学习重点
- 为什么 Flash Attention 能在不改变数值精度的前提下大幅加速？
- 掌握 AdamW 优化器中 Momentum 和 RMSProp 的结合逻辑。
- 理解量化带来的精度损失与显存收益的权衡。
