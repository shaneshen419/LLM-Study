# 🧠 LLM Learning Notes | 大模型学习笔记

![Build Status](https://img.shields.io/badge/Status-Learning-green) ![Last Update](https://img.shields.io/badge/Last%20Update-Jan%202026-blue) ![Topic](https://img.shields.io/badge/Topic-Deep%20Learning%20%26%20LLM-orange)

> 记录我个人学习 Large Language Models (LLM) 的核心知识点、算法原理及工程实践笔记。
> 内容涵盖底层 GPU 架构、Transformer 核心组件、训练/推理优化技术、分布式并行以及前沿模型分析（如 Qwen, DeepSeek, MoE 等）。

---

## 📖 目录 (Table of Contents)

主要分为以下五个核心模块：

- [📂 01. 基础架构与核心原理 (Basic Architecture)](./01_Basic_Architecture/)
    - Transformer
        - [整体架构](./01_Basic_Architecture/transformers/)
        - [Softmax](./01_Basic_Architecture/transformers/softmax.md)
    - Attention机制
        - [scaled_dot_product_attention](./01_Basic_Architecture/attention_mechanism/scaled_dot_product_attention/)
    - Positional Encoding
        - [Rotary Position Embedding](./01_Basic_Architecture/positional_encoding/rotary_position_embedding/)
    - BERT
- [📂 02. 训练与优化算法 (Training & Optimization)](./02_Training_Optimization/)
    - 优化器
    - Flash Attention
        - [flash attention](./02_Training_Optimization/flash_attention/)
    - 量化技术
- [📂 03. 分布式工程与显存优化 (Distributed Engineering)](./03_Distributed_Engineering/)
    - 并行策略：DP、TP、PP、SP、EP
        - [parallel strategy](./03_Distributed_Engineering/parallel_strategy/)
    - 显存优化：OOM解决方案
        - [agentic 长文本训练时候容易 oom，好的优化方案](./08_Interview_Experience/LLM/question/08.md)
        - [长上下文压缩有哪些方法？](./08_Interview_Experience/LLM/question/09.md)
    - SGLang：DP算法优化
    - vLLM：KV Cache、PageAttention、Prefix Cache
- [📂 04. 模型架构分析 (Model Analysis)](./04_Model_Analysis/)
    - Dense模型
    - MOE（混合专家）：稀疏激活原理、专家并行、门控网络与负载均衡
    - Qwen系列：模型结构与分词器设计
    - Deepseek：模型结构
- [📂 05. 前沿技术与对齐 (Advanced Topics)](./05_Advanced_Topics/)
    - RLHF对齐：奖励模型 (RM)、PPO、DPO、GRPO、GSPO 等算法原理。
        - [PPO(Proximal Policy Optimization)](./05_Advanced_Topics/ppo/)
        - [GDPO(Group reward-Decoupled Normalization Policy Optimization)](./05_Advanced_Topics/rlhf_alignment/GDPO/)
    - 思维链（CoT）：Long Chain-of-Thought 的机制与知识蒸馏
    - 推理：长上下文处理能力
- [📂 06. 应用开发与Agent生态 (Application & Agent Ecosystem)](./06_Application_Agent_Ecosystem/)
    - **RAG与数据增强**
        - LlamaIndex框架原理
        - 向量数据库 (Vector DB) 与索引策略
        - 文本向量化 (Embeddings) 与语义检索  <-- 你提到的"文本量化"如果是指文本转向量，放这里
    - **Agent架构与框架**
        - LangChain核心组件 (Chains, Memory)
        - Agent设计模式 (ReAct, Plan-and-Solve)
        - 多智能体协作 (Multi-Agent Systems, e.g., MetaGPT, AutoGen)
    - **协议与工具**
        - MCP (Model Context Protocol) 标准
        - Tool Use (Function Calling) 原理
- [📂 07. 思考与创新 (Thinking Space)](./07_Thinking_Space/)
- [📂 08. 面经（Interview Experience）](./08_Interview_Experience/LLM/README.md)
---

## ⏰后续计划
- Transformer架构各个模块的功能
- 几种并行策略（TP、PP...）
- 优化器原理
- RLHF的几种算法
- BERT、CLIP模型原理
- MOE、Dense模型
- vLLM架构原理：KV Cache、PageAttention、Prefix Cache
- SGLang架构原理
- Qwen1、2、3模型架构
- Deepseek模型架构
- Application Agent Ecosystem部分

## 🛠️ 关于笔记
本仓库笔记基于个人学习整理，参考了相关论文、官方文档及技术博客。如有错误，欢迎 Issue 指正。

## 📚 参考资料
*Wait to be updated...*

