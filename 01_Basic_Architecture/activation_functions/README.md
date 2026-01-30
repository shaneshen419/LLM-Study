# Activation Functions(激活函数)

## 一、SwiGLU 激活函数
SwiGLU 是目前主流大模型（如 Llama 2/3, Qwen, Deepseek, PaLM）在前馈神经网络（FFN）中普遍采用的激活结构。它由 Google 的 Noam Shazeer 在论文 “GLU Variants Improve Transformer” (2020) 中提出，被证明比传统的 ReLU 或 GELU 具有更好的收敛性和性能。

### 1.1 核心概念拆解
SwiGLU 并不是一个单一的数学函数，而是 Swish 激活函数 与 GLU (门控线性单元) 的组合体。

**a. Swish (SiLU) 激活函数**

Swish（也称为 SiLU, Sigmoid Linear Unit）是一个平滑的、非单调的激活函数。

$$
\text{Swish}(x)=x\cdot \sigma(\beta x)
$$

- 其中 $\sigma$ 是 Sigmoid 函数： $\sigma(z)=\frac{1}{1+e^{-z}}$ 。
- 通常 $\beta = 1$ ,此时就是SiLU： $x\cdot \sigma(x)$ 。
- **特点**：
    - **有下届无上届**：避免了梯度爆炸。
    - **非单调性**：在 $x$ 为负值区域有一个轻微的波谷，这有助于模型学习复杂的特性。

**b. GLU (Gated Linear Unit)**

GLU 借鉴了 LSTM/GRU 中的“门控”思想。它包含两个线性变换：一个负责传递信息（Content），一个负责控制流量（Gate）。

$$
\text{GLU}(x)=(x\cdot W)\odot \sigma (xV)
$$

- $xW$ ：是主要的信息变换。
- $xV$ ：经过Sigmoid归一化到（0，1）之间，作为一个“阀门”。
- $\odot$ ：按照元素相乘（Element-wise product）。
- **直观理解**：网路可以学习“这一层应该通过多少信息”，类似于注意力机制的简化版。

### 1.2. SwiGLUT 的数学定义
SwiGLU 将 GLU 中的 Sigmoid 门控替换为了 Swish 激活函数。

在 Transformer 的 FFN 层中，标准的 SwiGLU 结构如下：

$$
\text{FFN}_{\text{SwiGLU}}(x)=(\text{Swish}(xW_g)\odot (xW_{in}))W_{out}
$$

这里涉及三个权重矩阵（Standard ReLU 只有两个）：
- $W_g$ (Gate Projection)：计算门控信号，经过 Swish 激活。
- $W_{in}$ (Up Projection)：计算特征映射（不经过激活函数）。
- $W_{out}$ (Down Projection)：将门控后的结果映射回原始维度。

公式拆解：
- Gate path： $\text{Gate}=\text{SiLU}(x\cdot W_g)$ 
- Value Path: $Value=x\cdot W_{in}$ 
- Element-wise product: $H=\text{Gate}\odot \text{Value}$ 
- Output: $\text{Output}=H\cdot W_{out}$ 

### 1.3. 与标准FFN（ReLU）的对比

**a. 标准Transform FFN（ReLU）**

$$
\text{FFN}(x)=\text{ReLU}(xW_1 + b_1) W_2 + b_2
$$

- **结构**：先升维（通常是 $4d$ ），激活，再降维。
- **参数量**：2个矩阵（ $\text{dimes}4d$ 和 $4\text{dimes}d$ ） 

**b. SwiGLU FFN**

$$
\text{FFN}(x)=(\text{SiLU}(xW_g)\odot xW_{in})W_{out}
$$

- **结构**：有两条路径进行升维（Gate 和 Value），点积后再降维。
- **参数量**：3个矩阵。

**关键的工程调整（显存/参数对齐）**：

因为 SwiGLU 多了一个矩阵 （ $W_{in}$ 和 $W_g$ 都是升维），如果保持隐藏层维度为 $4d$ ，参数量会增加 50%。

为了保持参数量与标准Transformer 一致， **通常会将隐藏层维度由 $4d$ 减少到 $\frac{2}{3}\cdot4d \approx 2.68d$ 或 $\frac{8}{3}d$ 。**
- 例如 Llama2 的 7B 模型，Hidden dim 是4096，FFN的中间维度并没有设为 $4096\times 4=16384$ ,而是设为了 11008（接近 $4096\times \frac{8}{3}$ ）。

### 1.4. 为什么 SwiGLU 效果更好
- **更强的能力**：
    - 