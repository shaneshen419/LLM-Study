# 🖥️ Distributed Engineering | 分布式与工程架构

本模块解决“单卡放不下、算得慢”的问题，涵盖底层硬件架构、大规模并行策略及显存救急方案。

## 📝 核心内容

- **GPU 架构基础**
  - GPU 显存层级、计算单元与带宽瓶颈
- **大模型并行策略 (Parallelism)**
  - **DP** (Data Parallel)：数据并行
  - **TP** (Tensor Parallel)：张量并行
  - **PP** (Pipeline Parallel)：流水线并行
  - **SP** (Sequence Parallel) & **EP** (Expert Parallel)
  - 各并行策略的通信开销与适用场景对比
- **OOM (显存溢出) 解决方案**
  - 算法层面：Gradient Checkpointing (以时间换空间)
  - 工程层面：Offloading (卸载到CPU), Mixed Precision (混合精度)
  - SGLang 的 DP 算法优化

## 🎯 学习重点
- 在千亿参数模型训练中，如何组合使用 DP/TP/PP？
- 遇到 OOM 时，按照什么优先顺序排查和优化？
- 理解通信墙 (Communication Wall) 对分布式训练效率的影响。
