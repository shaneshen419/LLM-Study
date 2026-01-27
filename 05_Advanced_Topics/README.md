# 🚀 Advanced Topics | 对齐、推理与前沿技术

本模块记录大模型在预训练之后的高级技术，包括强化学习对齐、长上下文处理及推理能力增强。

## 📝 核心内容

- **训练全流程**
  - 预训练 (Pre-training) -> SFT (监督微调) -> RLHF (强化学习)
- **RLHF 与对齐算法**
  - **Reward Model**：奖励模型的构建与作用
  - **PPO** (Proximal Policy Optimization)
  - **DPO** (Direct Preference Optimization)：无奖励模型的偏好优化
  - **GRPO** & **DAPO**：新兴的优化策略
- **推理能力增强**
  - **CoT (Chain-of-Thought)**：长链思维机制解析
  - 知识蒸馏：领域知识蒸馏 vs 思维链蒸馏

## 🎯 学习重点
- 掌握 SFT 和 RLHF 分别赋予了模型什么能力？
- 理解 DPO 相比 PPO 在训练稳定性与资源消耗上的优势。
- CoT 是如何“涌现”出来的，以及如何通过蒸馏传递这种能力。
