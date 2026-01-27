# 🏗️ Basic Architecture | 基础架构与核心原理

本模块主要记录大模型（LLM）最底层的数学原理与网络架构设计。理解这些组件是掌握 LLM 的基石。

## 📝 核心内容

- **Transformer 完整架构**
  - Encoder 与 Decoder 的区别与联系
  - 宏观视角下的堆叠结构
  - 归一化层 (Normalization) 的位置与作用
- **Attention 机制深度解析**
  - Self-Attention 计算公式与推导
  - **Softmax**：为什么用幂指数？工程上的数值稳定性问题
  - Softmax 的 GPU 访存复杂度瓶颈分析
- **位置编码 (Positional Encoding)**
  - 绝对位置编码与相对位置编码
  - **RoPE (Rotary Position Embedding)**：旋转位置编码的数学推导与优势
  - ALiBi (Attention with Linear Biases)
- **经典模型回顾**
  - **BERT**：双向 Transformer 与 Embedding 模块设计

## 🎯 学习重点
- 理解 Attention 机制中 $Q, K, V$ 的物理含义。
- 掌握 RoPE 如何通过旋转矩阵注入位置信息。
- 理解 Softmax 在 GPU 计算中的内存带宽限制。
