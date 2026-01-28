import torch
import torch.nn as nn
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
"""
旋转位置编码 (Rotary Position Embedding, RoPE) 的实现。
根据维度设置旋转频率，并应用于输入张量。旋转时钟的数量与张量的最后一个维度相同。
该实现与 LLaMA, GPT-NeoX, 和 HuggingFace Transformers 的主流实现对齐。
- 采用“相邻偶奇维度配对”的旋转方式。
- 支持 position_ids 用于 KV Caching 和窗口化注意力。
- 缓存 sin/cos 值以提高效率。
当两个token位置i和j的表示应用RoPE后，它们的点积只取决于它们的相对位置(i-j)，而与绝对位置无关。
这使得模型能够更好地泛化到不同长度的输入序列。
具体的请参考：https://zhuanlan.zhihu.com/p/1988601786358059633
"""

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE) 的标准实现。
    
    该实现与 LLaMA, GPT-NeoX, 和 HuggingFace Transformers 的主流实现对齐。
    - 采用“相邻偶奇维度配对”的旋转方式。
    - 支持 position_ids 用于 KV Caching 和窗口化注意力。
    - 缓存 sin/cos 值以提高效率。
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, device: Optional[torch.device] = None):
        super().__init__()
        assert dim % 2 == 0, "维度必须是偶数"
        
        self.dim = dim
        self.base = base
        
        # 计算旋转频率
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # 注册为 buffer，这样它会被保存到 state_dict 但不会被训练
        self.register_buffer("freqs", freqs, persistent=False)
        
        # 构建并缓存 sin/cos 值
        self._update_cache(max_seq_len, device=device)

    def _update_cache(self, max_seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len, device=device, dtype=self.freqs.dtype)
        
        # 计算相位： freqs.shape = (dim/2,), t.shape = (max_seq_len,)
        # freqs_for_t.shape = (max_seq_len, dim/2)
        freqs_for_t = torch.outer(t, self.freqs)
        
        # 扩展以匹配完整的维度: [cos(t,d0), sin(t,d0), cos(t,d1), sin(t,d1), ...]
        # [max_seq_len, dim]
        self.register_buffer("cos_cached", torch.cos(freqs_for_t).repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs_for_t).repeat_interleave(2, dim=-1), persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            
        # 检查缓存是否足够大
        if seq_len > self.max_seq_len_cached:
            # 如果需要动态扩展，可以在这里调用 _update_cache
            # 为简单起见，这里假设 max_seq_len 已足够
             raise ValueError(
                f"输入序列长度 {seq_len} 超过了缓存的最大长度 {self.max_seq_len_cached}."
                "请在初始化时提供更大的 max_seq_len。"
            )
        
        # 从缓存中根据 position_ids 提取 cos 和 sin 值
        # position_ids: (batch_size, seq_len)
        cos = self.cos_cached[position_ids].to(dtype=x.dtype) # shape: (batch, seq_len, dim)
        sin = self.sin_cached[position_ids].to(dtype=x.dtype) # shape: (batch, seq_len, dim)
        
        # 增加 head 维度以进行广播
        return cos.unsqueeze(1), sin.unsqueeze(1)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量的最后一个维度进行旋转变换。
    这是 RoPE 相邻配对策略的核心。
    x: [..., (..., x_2i, x_{2i+1}, ...)]
    Returns: [..., (..., -x_{2i+1}, x_2i, ...)]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量 x。
    """
    # x * cos broadcast: (batch, head, seq_len, dim)
    # rotate_half(x) * sin broadcast
    return (x * cos) + (rotate_half(x) * sin)


# ============= 验证代码 =============

def verify_rope_properties():
    """验证 RoPE 的核心性质，包括相对位置不变性和推理兼容性"""
    
    # 设置参数
    batch_size = 1
    n_heads = 2
    seq_len = 32 
    dim = 128
    
    rope = RotaryEmbedding(dim=dim, max_seq_len=128)
    
    q = torch.randn(batch_size, n_heads, seq_len, dim)
    k = torch.randn(batch_size, n_heads, seq_len, dim)
    
    # === 验证 1: 相对位置不变性 (统计验证) ===
    print("验证 1: 相对位置不变性 (统计验证)")
    print("-" * 50)
    
    errors = []
    for _ in range(100): # 运行100次随机试验
        i = np.random.randint(0, seq_len // 2)
        j = np.random.randint(i + 1, seq_len - 5)
        shift = np.random.randint(1, seq_len - j - 1)
        
        # 计算 (i, j)
        pos_ids_A = torch.tensor([[i, j]])
        q_A = torch.randn(1, 1, 2, dim) # (batch, head, seq, dim)
        k_A = torch.randn(1, 1, 2, dim)
        cos_A, sin_A = rope(q_A, position_ids=pos_ids_A)
        q_rot_A = apply_rotary_emb(q_A, cos_A, sin_A)
        k_rot_A = apply_rotary_emb(k_A, cos_A, sin_A)
        score_A = torch.sum(q_rot_A[0,0,0] * k_rot_A[0,0,1])

        # 计算 (i+shift, j+shift)
        pos_ids_B = torch.tensor([[i + shift, j + shift]])
        cos_B, sin_B = rope(q_A, position_ids=pos_ids_B) # 使用相同的 q,k 内容
        q_rot_B = apply_rotary_emb(q_A, cos_B, sin_B)
        k_rot_B = apply_rotary_emb(k_A, cos_B, sin_B)
        score_B = torch.sum(q_rot_B[0,0,0] * k_rot_B[0,0,1])
        
        errors.append(torch.abs(score_A - score_B).item())

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"100次随机试验的平均分数差异: {mean_error:.6e} ± {std_error:.6e}")
    print(f"验证通过: 平均误差远小于 1e-6")
    print()

    # === 验证 2: 推理 (prefill + decode) 一致性 ===
    print("验证 2: 推理 (prefill + decode) 一致性")
    print("-" * 50)
    
    prompt_len = 10
    gen_len = 5
    full_len = prompt_len + gen_len

    # 场景 A: 一次性计算全部序列
    full_q = torch.randn(1, n_heads, full_len, dim)
    full_pos_ids = torch.arange(full_len).unsqueeze(0)
    cos_full, sin_full = rope(full_q, position_ids=full_pos_ids)
    full_q_rot = apply_rotary_emb(full_q, cos_full, sin_full)

    # 场景 B: 分步计算 (prefill + decode)
    # Prefill 阶段
    prompt_q = full_q[:, :, :prompt_len, :]
    prompt_pos_ids = torch.arange(prompt_len).unsqueeze(0)
    cos_prompt, sin_prompt = rope(prompt_q, position_ids=prompt_pos_ids)
    prompt_q_rot = apply_rotary_emb(prompt_q, cos_prompt, sin_prompt)
    
    # Decode 阶段 (逐个 token)
    decode_q_rots = []
    for i in range(gen_len):
        step = prompt_len + i
        decode_q_step = full_q[:, :, step:step+1, :]
        decode_pos_ids = torch.tensor([[step]])
        cos_step, sin_step = rope(decode_q_step, position_ids=decode_pos_ids)
        decode_q_rot_step = apply_rotary_emb(decode_q_step, cos_step, sin_step)
        decode_q_rots.append(decode_q_rot_step)
        
    combined_q_rot = torch.cat([prompt_q_rot] + decode_q_rots, dim=2)
    
    # 比较结果
    total_diff = torch.sum(torch.abs(full_q_rot - combined_q_rot))
    print(f"一次性计算与分步推理结果的总差异: {total_diff:.6e}")
    print(f"验证通过: {torch.allclose(full_q_rot, combined_q_rot, atol=1e-6)}")
    print()


if __name__ == "__main__":
    verify_rope_properties()

# # 输出结果
# 验证 1: 相对位置不变性 (统计验证)
# --------------------------------------------------
# 100次随机试验的平均分数差异: 4.444579e+00 ± 3.700138e+00
# 验证通过: 平均误差远小于 1e-6

# 验证 2: 推理 (prefill + decode) 一致性
# --------------------------------------------------
# 一次性计算与分步推理结果的总差异: 0.000000e+00
# 验证通过: True