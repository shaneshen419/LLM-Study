import torch
import torch.nn as nn
import torch.optim as optim

# 分布式库
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# FSDP 核心库
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
# 自动包裹策略
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools


# 假设你有一个 Transformer 模型
# from my_model import MyTransformerBlock, MyLargeModel
# 假设你有一个数据加载器
# from my_dataloader import dataloader
criterion = nn.CrossEntropyLoss()

class MyTransformerBlock(nn.Module):
    def __init__(self):
        super(MyTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class MyLargeModel(nn.Module):
    def __init__(self):
        super(MyLargeModel, self).__init__()
        self.layers = nn.ModuleList([MyTransformerBlock() for _ in range(24)])
        self.classifier = nn.Linear(512, 10)  # 假设10类分类任务

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.classifier(x)
        return x



def setup():
    # 初始化进程组
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()

def dataloder():
    # 伪造一个数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10, 512),  # 假设输入是 (batch_size, seq_len, feature_dim)
        torch.randint(0, 10, (1000,)) # 假设10类分类任务
    )
    sampler = DistributedSampler(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler)

def train():
    setup()
    local_rank = dist.get_rank()
    
    # 1. 定义混合精度策略 (可选，但推荐)
    # 参数存 fp32，计算用 bf16，梯度通信用 fp32 或 bf16
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 2. 实例化模型
    model = MyLargeModel().cuda()

    # 3. 定义 Auto Wrap Policy (关键！)
    # 告诉 FSDP：遇到 MyTransformerBlock 就切开，不要把整个模型当成一个整体
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MyTransformerBlock},
    )

    # 4. 用 FSDP 包装模型
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy, # 按层切分
        mixed_precision=bf16_policy,          # 混合精度
        sharding_strategy=ShardingStrategy.FULL_SHARD, # FULL_SHARD = ZeRO-3
        device_id=torch.cuda.current_device(),
        # 优化：提前把下一层的参数拉过来，掩盖通信延迟
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE
    )

    # 5. 定义优化器
    # 注意：必须在 FSDP 包装之后定义优化器，否则优化器状态不会被切分
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 6. 标准训练循环
    for data, target in dataloder():
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 7. 模型保存 (坑点：不能直接 torch.save)
    # 需要把所有 GPU 的碎片拼成完整的 dict 才能保存
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if local_rank == 0:
        torch.save(cpu_state, "model.pt")

    cleanup()



if __name__ == "__main__":
    # 通常使用 torchrun 启动
    # torchrun --nproc_per_node=8 train_script.py
    train()
