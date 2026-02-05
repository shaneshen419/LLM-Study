import torch
import torch.distributed as dist
import torch.nn as nn

class ExpertParallelLayer(torch.nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim, world_size):
        super().__init__()
        # 假设每个 GPU 只负责 1 个 Expert
        self.my_expert_id = dist.get_rank()
        # 本地只初始化这就一个 Expert 的权重，大大节省显存！
        self.expert_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.router = nn.Linear(input_dim, num_experts)
        self.world_size = world_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim] -> flatten -> [total_tokens, dim]
        tokens = x.view(-1, x.size(-1))
        
        # 1. Router 计算
        logits = self.router(tokens)
        probs = torch.softmax(logits, dim=-1)
        # 选出 Top-1 (简化演示) 专家索引
        max_prob, dest_expert_indices = torch.max(probs, dim=-1) 
        
        # 2. 调度准备 (Dispatch Preparation)
        # 计算每个 Token 要去哪个 Rank (假设 Expert ID = Rank ID)
        # 需要按照去往的 Rank 对 Token 进行排序，才能做 All-to-All
        sort_indices = torch.argsort(dest_expert_indices)
        tokens_sorted = tokens[sort_indices]
        
        # 计算每个 Rank 需要接收多少个 Token (为了 Padding 或 Split)
        # 这里省略了复杂的 capacity limiting 和 padding 逻辑
        # splits 记录了发给 Rank 0, Rank 1... 分别有多少数据
        send_splits = [sum(dest_expert_indices == r) for r in range(self.world_size)]
        
        # 3. 第一次通信: Dispatch (All-to-All)
        # 将 Token 发送到对应的专家 GPU
        # input_splits 是我发给别人的，output_splits 是别人发给我的
        received_tokens = dist.all_to_all_single(tokens_sorted, output_split_sizes=..., input_split_sizes=send_splits)
        
        # 4. 专家计算 (Computation)
        # 此时 received_tokens 都是发给我这个 Expert 的
        expert_output = self.expert_mlp(received_tokens)
        
        # 5. 第二次通信: Combine (Reverse All-to-All)
        # 把计算结果发回原 GPU
        final_output_sorted = dist.all_to_all_single(expert_output, output_split_sizes=send_splits, input_split_sizes=...)
        
        # 6. 恢复原始顺序
        # 因为之前 argsort 乱了顺序，现在要还原回去
        final_output = torch.empty_like(tokens)
        final_output[sort_indices] = final_output_sorted
        
        # 7. 门控加权
        result = final_output * max_prob.unsqueeze(1)
        
        return result.view(x.shape) # 还原形状
