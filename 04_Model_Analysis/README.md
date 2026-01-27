# 🔍 Model Analysis | 模型架构分析与对比

本模块深入剖析当前主流的开源大模型架构，对比其设计差异与技术选型。

## 📝 核心内容

- **MoE (Mixture of Experts) 混合专家模型**
  - 稀疏激活原理：Gating Network (门控) 与 Top-K 路由
  - 专家并行 (Expert Parallelism)
  - MoE vs Dense：计算效率、推理时延与性能对比
  - 训练难点：负载均衡 (Load Balancing) 问题
- **Qwen (通义千问)**
  - 模型结构特性
  - Tokenizer 分词原理
- **DeepSeek vs Qwen**
  - 深度横向对比：模型规模、效率、开源策略
  - 推理能力强化方案

## 🎯 学习重点
- 理解 MoE 如何实现“参数量巨大但推理计算量小”。
- 分析 DeepSeek 与 Qwen 在架构细节上的微小差异如何影响最终性能。
