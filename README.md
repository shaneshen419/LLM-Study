# 🧠 LLM Learning Notes | 大模型学习笔记

![Build Status](https://img.shields.io/badge/Status-Learning-green) ![Last Update](https://img.shields.io/badge/Last%20Update-Jan%202026-blue) ![Topic](https://img.shields.io/badge/Topic-Deep%20Learning%20%26%20LLM-orange)

> 记录我个人学习 Large Language Models (LLM) 的核心知识点、算法原理及工程实践笔记。
> 内容涵盖底层 GPU 架构、Transformer 核心组件、训练/推理优化技术、分布式并行以及前沿模型分析（如 Qwen, DeepSeek, MoE 等）。

---

## 📖 目录 (Table of Contents)

主要分为以下五个核心模块：

- [📂 01. 基础架构与核心原理 (Basic Architecture)](./01_Basic_Architecture/)
    - Transformer
    - Attention机制
    - Positional Encoding
        - [Rotary Position Embedding](./01_Basic_Architecture/rotary_position_embedding/)
    - BERT
- [📂 02. 训练与优化算法 (Training & Optimization)](./02_Training_Optimization/)
    - 优化器
    - Flash Attention
    - 量化技术
- [📂 03. 分布式工程与显存优化 (Distributed Engineering)](./03_Distributed_Engineering/)
    - 并行策略：DP、Tp、PP、SP、
    - 显存优化：OOM解决方案
    - SGLang：DP算法优化
    - vLLM：KV Cache、PageAttention、Prefix Cache
- [📂 04. 模型架构分析 (Model Analysis)](./04_Model_Analysis/)
    - Dense模型
    - MOE（混合专家）：稀疏激活原理、专家并行、门控网络与负载均衡
    - Qwen系列：模型结构与分词器设计
    - Deepseek：模型结构
- [📂 05. 前沿技术与对齐 (Advanced Topics)](./05_Advanced_Topics/)
    - RLHF对齐：奖励模型 (RM)、PPO、DPO、GRPO 等算法原理。
    - 思维链（CoT）：Long Chain-of-Thought 的机制与知识蒸馏
    - 推理：长上下文处理能力

---

## 🛠️ 关于笔记
本仓库笔记基于个人学习整理，参考了相关论文、官方文档及技术博客。如有错误，欢迎 Issue 指正。

## 📚 参考资料
*Wait to be updated...*

