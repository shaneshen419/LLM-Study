import torch
import math
"""
参考文章：https://zhuanlan.zhihu.com/p/28992140997
实现缩放点积注意力机制 (Scaled Dot-Product Attention, SDPA)。
GQA (分组查询注意力) 和 因果掩码 (Causal Masking)
    1. GQA: 通过将查询分组来减少计算量，同时保持性能。
        a: 对齐key和value的序列长度,以适应分组查询的需求。
        b: 通过 repeat_interleave 方法扩展 key 和 value 的序列长度，使其与 query 对齐。
    2. 因果掩码: 确保每个位置只能关注其之前的位置，适用于自回归模型。
    3. 缩放机制：将 Query (Q) 和 Key (K) 的点积结果除以 √dk（即 query.size(-1) 的平方根）。
        a: 主要是为了防止 softmax 进入饱和区，从而避免梯度消失。
        b: 如果不加入缩放因子，随着 dk 维度的增加，点积的数值范围会变得很大，输入 softmax 后的分布会变得十分尖锐，就像 one-hot 编码一样。
        c: 函数曲线变得极其平缓，容易梯度消失。
        d: 加入缩放因子之后 softmax 的结果分布会更加平滑，梯度能够正常回传。
    4. 因果掩码Causal Masking：
        a: 在自回归模型中，确保每个位置只能关注其之前的位置，防止信息泄露。
        b: 通过创建一个上三角矩阵作为掩码，将未来位置的注意力权重设置为负无穷大，从而在 softmax 后变为零。
"""
def scaled_dot_product_attention(query, key, value, 
                                 attn_mask=None, 
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=None,
                                 enable_gqa=False) -> torch.Tensor:
    # 初始化与缩放因子
    Q_sequencelen, K_sequencelen = query.size(-2), query.size(-3)
    # 计算缩放因子1/√d_k
    """
        缩放因子主要是为了防止softmax进入饱和区，从而避免梯度消失。
        如果不加入缩放因子，随着dk维度的增加，点积的数值范围会变得很大，输入softmax后的分布会变得十分尖锐。就像one-hot编码一样。
        函数曲线变得极其平缓，容易梯度消失。
        加入缩放因子之后softmax的结果分布会更加平滑，梯度能够正常回传。
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # 初始化atten_bias
    attn_bias = torch.zeros(Q_sequencelen, K_sequencelen, dtype=query.dtype, device=query.device)

    # 处理因果掩码
    if is_causal:
        assert attn_mask is None
        # 创建上三角掩码
        temp_mask = torch.ones(Q_sequencelen, K_sequencelen, device=query.device, dtype=torch.bool).triu(diagonal=0)
        # 仅保留上三角部分
        attn_bias.masked_fill_(temp_mask.logical_not(), float('-inf'))
        # 返回与query相同数据类型的attn_bias
        attn_bias.to(query.dtype)
    
    # 处理gqa
    if enable_gqa:
        # 对齐key和value的序列长度
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

    # 计算注意力矩阵
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # 添加注意力偏置
    attn_weight += attn_bias
    # softmax归一化
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # dropout: 仅在训练时开启，推理时通常关闭。
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, train=True)
    return attn_weight @ value
        
