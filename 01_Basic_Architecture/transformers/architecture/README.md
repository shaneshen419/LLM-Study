# Transformer架构
这部分主要介绍Transformer架构的：self-Attention机制、多头注意力(MHA、MQA、GQA)、掩码自注意力、交叉注意力(cross attention)、位置编码(Embedding)、前馈网络(FFN)、残差连接、层归一化(Batch Norm、Layer Norm、RMSNorm)各自的功能。

## 一、self-Attention机制

### 1.1. Self-Attention机制

- **功能**：让模型在处理当前词（Token）时，能够“看到”句子中其他所有词，并根据相关性聚合信息。解决了长距离依赖问题。
- **原理**：将输入向量映射为三个向量：Query (查询), Key (键), Value (值)。
    - **Query(Q)**：当前词在寻找什么信息。
    - **Key(K)**：其他词包含什么特征。
    - **Value(V)**：其他词的具体内容。
    - 通过计算 $Q$ 和 $K$ 的点积来确定相关性（注意力分数），归一化后作为权重加权求和 $V$ 。
- **公式**：
    - $\sqrt{d_k}$ 是缩放因子，防止点积过大导致 Softmax 梯度消失。

$$
\text{Attention}=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

### 1.1. Softmax的作用

## 二、多头注意力(MHA、MQA、GQA)
这一块是现代大模型优化的重点（显存 vs 性能）。

### a. Multi-Head Attention (MHA - 标准多头注意力)

- **功能**：让模型从不同的“子空间”关注信息（例如：一个头关注语法结构，另一个头关注语义指代）。
- **原理**：将 $d_{model}$ 维度的向量拆分为 $h$ 个头，每个头独立计算 Attention，最后拼接。
- **公式**：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\text{head}_2,\cdots,\text{head}_h)W^O
$$
$$
\text{head}_i = \text{Attention}(QW^Q_i,QW^K_i,QW^V_i)
$$

- **特点**： $Q,K,V$ 的头数相同 $(H_q=H_k=H_v)$ 。显存占用最大。

### b. Multi-Query Attention (MQA)
- **功能**：极大减少推理时的显存占用（KV Cache），提高生成速度。
- **原理**：所有的 Query 头**共享**同一组 Key 和 Value 头。
- **特点**： $H_q=h,H_k=1,H_v=1$ 。性能会有轻微损失，因为这是有损压缩，但推理极快。

### c. Grouped-Query Attention (GQA - 分组查询注意力)
- **功能**：MHA 和 MQA 的折中方案（Llama 2/3, Qwen 采用）。
- **原理**：将 Query 分成 $g$ 组，每组共享一对 $K$ , $V$ 。
- **特点**： $H_q=h,H_k=g,H_v=g$ 其中（ $1<g<h$ ）。在保持效果的同时显著降低显存。同时也是有损压缩，一组Q共用同一个KV，会导致部分的性能有所损失。

## 三、Masked Self-Attention (掩码自注意力)
- **功能**：主要用于 Decoder（如 GPT 类模型）。防止模型在预测当前词时“偷看”到后面的词。
- **原理**：在 Attention 分数矩阵计算 Softmax 之前，将矩阵上三角区域（即未来位置）的值设为负无穷大 ( $-\infty$ )。
- **公式**：

$$
\mathbf{M}_{ij} = 
\begin{cases} 
0 & 	\text{if }\quad i \ge j \\
-\infty & 	\text{if }\quad i < j
\end{cases}
$$
$$
\text{MaskAttention}=\text{softmax}(\frac{QK^T}{\sqrt{d_k}}+M)V
$$

## 四、交叉注意力(cross attention)
- **功能** ：用于 Encoder-Decoder 架构（如原始 Transformer, T5）。连接编码器和解码器。
- **原理** ：让解码器生成的序列能够利用编码器输入的源信息。
- **区别** ：
    - **Query**( $Q$ )来自Decoder的上一层输出。
    - **Key**( $K$ ) & Value( $V$ )来自Encoder的最终输出。 

## [五、位置编码(Embedding)请参考](../../positional_encoding/)

## 六、前馈网络(FFN / MLP)
- **功能**：Transformer 中的“记忆”区域和非线性变换核心。Attention 负责“路由信息”，FFN 负责“处理和存储知识”。
- **原理**：两个线性变换中间夹一个激活函数。现在主流模型（如 Llama, Deepseek）常用 SwiGLU 替代标准的 ReLU。
- **公式(标准)**：

$$
\text{FFN}(x)=\text{max}(0,xW_1+b_1)W_2+b_2
$$

- **公式（SwiGLU变体）**

$$
\text{FFN}(x)=(\text{Swish}(xW_G)\odot (xW_1))W_2
$$

## 七、Residual Connections (残差连接)
- **功能**：解决深层网络中的梯度消失问题，允许训练极深的模型。
- **原理**：将输入 $x$ 直接加到子层的输出上。
- **公式**：

$$
x_{output}=\text{LayerNorm}(x+\text{SubLayer}(x))
$$

(注：这是 Post-Norm 写法，现在的 LLM 多用 Pre-Norm，即先 Norm 再进子层)

## 八、归一化技术 (Normalization)

这是让模型稳定训练的关键。

### a. Batch Normalization (BN)
- **原理**：对**整个 Batch** 的样本在同一维度做归一化。
- **缺点**：不适合 NLP，因为句子长度不一且 Batch size 较小时统计不准。Transformer 基本不用。

### b. Layer Normalization (LN)
- **原理**：对**单个样本**的所有特征维度计算均值和方差进行归一化。不依赖 Batch size。
- **公式**：

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma ^2+\varepsilon}}\cdot \gamma + \beta
$$

### c. RMSNorm (Root Mean Square Norm)
- **功能**：Layer Norm 的简化版，目前最主流（Llama, Gema, Deepseek 都在用）。
- **原理**：去掉了减均值（ $\mu$ ）的操作，只保留缩放。计算量更小，效果相当甚至更好。
- **公式**：

$$
\text{RMSNorm}(x)=\frac {x}
{\sqrt {\frac {1}{d} \sum ^d_{i=1} x^2_i + \varepsilon}}\cdot \gamma
$$