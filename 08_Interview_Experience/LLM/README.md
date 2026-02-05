# 这部分是大模型的面经总结
这部分记录大模型的面试总结，包括自己的面试问题和网上大佬公布的大厂的面试经历。

## 一、美团大模型面经-2026-01-28
这部分是一个大佬公布他的美团的大模型的面试经历的内容，做了简单的总结，结合LLM和网上的资料总结了一下答案
### question
```python
# 一面 
1. MOE 原理，讲一下负载均衡以及有哪几种类型？ 
2. 你能写一下公式吗？ MOE 的 expert parallel 如何做的？ 
3. 你用了 swift 参数怎么设置的，有 2 个 node 你如何分配你的训练参数？ 
4. 做 grpo 遇到熵崩没有？你是怎么理解的？
5. 如何解决的（clip higher 限制、小学习率） 大模型容易 reward hacking，如何解决？
6. 你说训一个小模型，那小模型数据如何来？还有什么办法吗？（这里我多回复了关键词定位来判断是否有 reward hacking 或者强制要求设置 step1XXX，然后最后根据关键词判断思维链和 answer 是否对不对，否则只有 llm as a judge 了） 
7. VLLM prefix cache实现过吗？ 讲一下你的理解
8. 在训练 grpo 时候应该更新rollout 的 mllm 吗？（ref model） 
9. 手撕： transformers encoder 

# 二面 
1. 如何理解 dspo 这几个算法的创新？（这里有点难理解，不过还是回答了从 grpo 到 dapo 到 dspo 的几个思路） 
2. agentic 长文本训练时候容易 oom，你有什么好的优化吗？ （外挂 db 做 agentic 上下文检索，不过对方不太满意，觉得太复杂了） 
3. 长上下文压缩有哪些方法？ 
4. 手撕： k 个链表反转、岛屿面积 

# 三面 
1. 更多的是对他们某一个业务做了一个交流和对大模型的看法，老板觉得做 cot 目前没有什么大的创新（确实啊），然后还说组内也在讨论做具身（感觉现在视觉组都会开始考虑具身方向了），最后问了我对具身的了解。
```

## 1. MOE 原理，讲一下负载均衡以及有哪几种类型？

### （a）MOE的核心原理
MOE是为了在FLOPs不变的情况下扩大参数量。
- **Dense模型（稠密模型）**：对于每一个输入的Token，网络中的**所有**参数都要参与计算。
- **Sparse MeE模型（稀疏模型）**：将网络中的某些层（通常是MLP层）替换为一组“专家”（Experts）。对于每一个Token，只有一个“路由器”（Router/Gate）决定它去哪几个专家那里进行计算。

**核心公式**

$$
y=\sum^{N}_{i=1}G(x)_i\cdot E_i(x)
$$

- $x$ ：输入Token。
- $E_i(x)$ ：第 $i$ 个专家的输出（通常是一个 Feed-Forward Network）。
- $G(x)_i$ ：门控网络（Router）给第 $i$ 个专家的权重（概率），通常是稀疏的（Top-K，例如只取前 2 个，其余为 0）。

**优势**
- **参数量巨大**：可以把模型做到万亿参数。
- **计算量（FLOPs）小**：因为每次只激活很少一部分专家，推理和训练速度快。

### （b）为什么要负载均衡？
这是 MoE 训练中最棘手的问题。如果没有干预，Router 很容易导致**两个严重后果**：
- **训练坍塌（Collapse）/赢者通吃（Winner-takes-all）
    - Router 可能会发现某一个专家（比如 Expert 0）初始化稍微好一点点。
    - Router 就拼命把所有 Token 都发给 Expert 0。
    - Expert 0 得到的训练数据最多，变得越来越强；其他专家（Expert 1~N）没数据吃，越来越弱。
    - **结果**：MoE 退化成了一个单体的小模型，浪费了其他专家的参数。
- **计算瓶颈 / 显存溢出**
    - 在分布式训练中，不同的专家通常放在不同的GPU上。
    - 如果 Token 分配不均，GPU 0（负责热门专家）会很忙，显存爆满；GPU 1（负责冷门专家）在空转（Idle）。
    - **木桶效应**：整个系统的速度取决于最慢的那个 GPU，导致训练效率极低。

### （c）负载均衡的类型与策略
为了解决上述的问题，工业界发展出了多种策略，主要分为“**基于辅助损失的软约束**”和“**基于容量的硬约束**”，以及最新的“**架构级创新**”。

**类型一：辅助损失（Auxiliary Loss / Load Balancing Loss）**

这是最经典的方法（Google Switch Transformer，GShard，Mixtral都在用）。
- **原理**：在总Loss中加一项惩罚项，如果专家接收的Token数量方差太大，Loss就变大。
- **公式概念**：

    $$
    Loss_{total}=Loss_{task}+\alpha \cdot Loss_{load-balance}
    $$

    通常 $Loss_{load-balance}=N\cdot \sum^N_{i=1}f_i\cdot P_i$ 

    - $f_i$ ：第 $i$ 个专家实际接收到的 Token 比例。
    - $P_i$ ：Router预测分配给第 $i$ 个专家的平均概率。
    - 我们希望 $f$ 和 $P$ 都是均匀分布的（即 $1/N$ ），此时点积最小。
- **优点**：实现简单，不强制丢弃数据。
- **缺点**：超参数 $\alpha$ 难调。如果太大，会干扰主任务训练；如果太小，不起作用。

**类型二：容量限值与 Token 丢弃**

这是一种**硬约束**，通常配合辅助损失使用。
- **原理**：
    - 设定一个 **Capacity Factor（容量因子，C）**。比如 $C=1.1$ ，意味着每个专家最多处理“平均值 $\times 1.1$ ” 个 Token。
    - 假设一个 Batch 有100个 Token，有 4 个专家。平均每个专家处理 25 个。设定 $C=1.2$ ，则上限为 30 个。
    - **Token Dropping**：如果第 31 个Token 也被路由到了该专家，由于缓冲区满了，这个Token会被**直接丢弃**（不进行MLP计算，直接通过残差连接跳过，或者计算结果置零）。
- **优点**：严格保证了 GPU 的显存和计算负载是均衡的，工程实现非常稳定。
- **缺点：丢弃 Token 会损失模型性能**。模型看书看到一半跳过了几个字，肯定不好。

**类型三：Router Z-Loss**

- **原理**：在Logits 进入Softmax 之前，限值其数值大小。
- **目的**：防止Router过于自信（输出的Logit极大），导致梯度消失，Router 无法更新策略。这是一种稳定训练的手段，间接帮助负载均衡。

**类型四：Expert Choice Routing（专家选Token）**

这是 Google 提出的颠覆性思路（反客为主）。
- **原理**：
    - **传统**：Token 选 Top-K 个专家。
    - **Expert Choice：专家选 Top-k个 Token**。
    - 每个专家查看 Batch 里所有Token，挑选自己最擅长（Logit最高）的那 k 个。
- **优点**：**天然完美负载均衡**。因为每个专家设定的 k 是一样的，大家干的活绝对一样多。
- **缺点**：
    - 有的Token可能被很多专家选中（过度计算）。
    - 有的Token可能一个专家都没选中（被迫丢弃/只有残差）。

**类型五：无负载均衡 / 共享专家（DeepSeek-V2/V3等新架构）**

最新的趋势是尽量减少对 Loss 的强行干扰，通常架构设计来平衡。
- **Shared Expert（共享专家）**：
    - 除了 Routed Experts（选来选去的专家），设立一个或多个**Shared Experts**。
    - **所有 Token**都会经过 Shared Experts。
    - **作用**：Shared Experts 负责捕获“通用尝试”（语法、常识），Router Experts 负责“专业知识”。这样即使 Router 发生倾斜，Shared Experts也能兜底。
- **Bias Correction（偏置修正）**：
    - 不直接加 Loss，而是在推理时动态调整 Router的Bias，强行降低热门专家的被选概率（仅在推理时或训练特定阶段使用）。
- **Auxiliary-Loss-Free**：
    - DeepSeek-V2 提出了一种改进的负载均衡 Loss，甚至在某些阶段尝试去除Loss，通过 Group-limited Routing（分组路由）来限值搜索空间，从而自然平衡。

>在面试中，回答这个问题的逻辑链条应该是：
**定义**：MoE 是为了在 FLOPs 不变的情况下扩大参数量。\
**问题**：核心挑战是 Router 分配不均导致的“坍塌”和“GPU 计算瓶颈”。\
**解决方案**：\
（1）Soft (辅助 Loss)：最常用，加个惩罚项让 Router 均匀发牌。\
（2）Hard (容量限制)：强制截断，满了就 Drop Token（工程必备，但要尽量避免触发）。\
（3）Novel (架构创新)：Expert Choice（专家选人）或 Shared Experts（DeepSeek 模式，通用+专用分离）。

## 2. MOE的公式是什么？ MOE 的 expert parallel 如何做的？

### （a）MOE 的核心数学公式
假设输入 Token 的向量为 $x\in \bold R^d$ ( $d$ 为 hidden dimension)。

模型中有 $N$ 个专家网络 $\{E_1,E_2,\dots,E_n\}$ ,通畅 $E_i$ 就是一个 FFN。

- **输出公式**

    MOE层的输出 $y$ 是被选中的专家输出的**加权和**：

    $$
    y=\sum^N_{i=1}G(x)_i\cdot E_i(x)
    $$

    其中 $G(x)$ 是 Gating Network（Router）的输出向量，表示每个专家的权重。

- **门控网络与 Top-k**

    为了保持稀疏性，我们不会计算所有 $N$ 个专家，而是只取 Top-k（通常 $k-2$ ）。

    首先计算原始路由得分（Logits）：

    $$
    h(x)=x\cdot W_g
    $$

    其中 $W_g\in \bold R^{d\times N}$ 是 Router 的可学习权重矩阵。

    然后进行 **Top-k Softmax**处理：
    - 找出 $h(k)$ 中数值最大的 $k$ 个索引，记为集合 $\varGamma$ 。
    - 对于不属于 Top-k 的专家，权重置为 0；对于属于 Top-k 的专家，做归一化：

    $$
    G(x)_i=
    \left\{
        \begin{array}{rcl}
        \frac{e^{h(x)_i}}{\sum_{j\in \varGamma}e^{h(x)_j}}\quad\text{if}\quad{i\in \varGamma} \\
        0\quad \text{otherwise}
        \end{array}
    \right.
    $$

- **负载均衡辅助损失**

    为了防止训练坍塌，通常加入辅助 Loss：

    $$
    L_{aux}=\alpha \cdot N \cdot \sum^N_{i=1}f_i\cdot P_i
    $$

    - $f_i$ ：这个 Batch 中实际被分配给专家 $i$ 的 Token比例（离散统计）。
    - $P_i$ ：Router 预测分配给专家 $i$ 的平均概率（连续值，即 $\text{softmax}(hx)$ 的平均）。
    - 目标是最小化 $f\cdot P$ ，即让分布尽量接近均匀分布 $1/N$ 。

### （b）Expert Parallelism（EP）是如何做的？
在传统的 Data Parallel（DP）中，每个 GPU 复制模型，处理不同的数据。

在 Expert Parallel（EP）中，**不同的专家驻留在不同的GPU上**。

假设我们有 4 个GPU（Rank 0~3），模型有 4 个专家（E0~E3）。
- GPU 0 持有 Expert 0 的权重。
- GPU 1 持有 Expert 1 的权重
- ...

**核心流程：All-to-All通信**
    
EP的本质是**数据的重排与迁移**。

1. **步骤1：路由计算**

    每个GPU拿到一个Batch的数据（比如 Batch size=4）。GPU 0 算出它的 4 个Token 分别要去哪：
    - Token A -> 需去 Expert 1（在GPU 1）
    - Token B -> 需去 Expert 0（在GPU 0，本地）
    - Token C -> 需去 Expert 3（在GPU 3）
    - Token D -> 需去 Expert 1（在GPU 1）
2. **步骤2：第一次通信（Dispatch / All-to-all）**

    所有 GPU 将手中的Token 发送到对应的“专家所在的GPU”。

    这是一次 **All-to-All** 操作（每个人都要给其他人发东西，每个人也要收东西）。
    - GPU 0 把 Token A,D 发给 GPU 1.
    - GPU 0 把 Token C 发给 GPU 3.
    -（同事，GPU 0也会收到来自 GPU 1,2,3 发给 Expert 0的Token）。

3. **步骤3：专家计算**

    现在，GPU 1 拥有了来自所有 GPU 的、需要 Expert 1 处理的 Token。

    GPU 1 运行 Expert 1（FFN）进行计算。

    $$
    Y_{local}=\text{FFN}_1(X_{received}) 
    $$

4. **步骤4：第二次通信（Combine / All-to-All）**

    计算完了，得把结果还给“原主”。

    再次进行 **All-to-All**，将计算结果逆向发回给 Token 原始所在的 GPU。

5. **步骤5：结果加权**

    GPU 0 收到了 Token A,B,C,D 处理后的结果，乘以对应的门控权重 $G(x)$ ，通过残差连接输出。

### （c）EP 的伪代码实现 (PyTorch 风格)
[**EP** 的代码点击这里](./ExpertParallel.py)

---
---

## 3. 你用了 swift 参数怎么设置的，有 2 个 node 你如何分配你的训练参数？ 
