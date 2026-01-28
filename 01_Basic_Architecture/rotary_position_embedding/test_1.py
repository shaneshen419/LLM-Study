import torch
import torch.nn.functional as F
import numpy as np

# --- 1. 环境设置 ---
# 为了结果可复现，设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义词汇表和词向量 (Embedding)
vocab = {"<pad>": 0, "人": 1, "咬": 2, "狗": 3}
# d_model = 8 (嵌入维度)
embeddings = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=8)

# 两个语义完全不同的句子
sentence1 = ["人", "咬", "狗"]
sentence2 = ["狗", "咬", "人"]

# 将句子转为 token ID
tokens1 = torch.tensor([vocab[word] for word in sentence1]) # tensor([1, 2, 3])
tokens2 = torch.tensor([vocab[word] for word in sentence2]) # tensor([3, 2, 1])

# 获取词向量
x1 = embeddings(tokens1) # 句子1的输入矩阵 X
x2 = embeddings(tokens2) # 句子2的输入矩阵 X

# --- 2. 模拟 Attention 计算 ---
d_model = 8  # 输入维度
d_k = 16     # Q, K 维度
d_v = 16     # V 维度

# 定义共享的权重矩阵 Wq, Wk, Wv
W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_v)

# 定义一个函数来执行 Attention 计算
def scaled_dot_product_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V
    return output

# --- 3. 计算并对比结果 ---
output1 = scaled_dot_product_attention(x1, W_q, W_k, W_v)
output2 = scaled_dot_product_attention(x2, W_q, W_k, W_v)

print("句子1 ('人', '咬', '狗') 的 Attention 输出:")
print(output1)
print("\n句子2 ('狗', '咬', '人') 的 Attention 输出:")
print(output2)

print("\n--- 结果分析 ---")
# 提取 “人” 在两个句子中的输出向量
# 在句子1中，“人”是第0个词
output_ren_1 = output1[0] 
# 在句子2中，“人”是第2个词
output_ren_2 = output2[2]

# 提取 “狗” 在两个句子中的输出向量
# 在句子1中，“狗”是第2个词
output_gou_1 = output1[2]
# 在句子2中，“狗”是第0个词
output_gou_2 = output2[0]

# 检查向量是否相等
are_ren_vectors_equal = torch.allclose(output_ren_1, output_ren_2)
are_gou_vectors_equal = torch.allclose(output_gou_1, output_gou_2)

print(f"“人”在两个句子中的输出向量是否相同? {are_ren_vectors_equal}")
print(f"“狗”在两个句子中的输出向量是否相同? {are_gou_vectors_equal}")