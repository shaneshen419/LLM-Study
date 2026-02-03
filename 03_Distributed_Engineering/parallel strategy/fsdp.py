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

# 假设你有一个 Transformer 模型
from my_model import MyTransformerBlock, MyLargeModel

def setup():
    # 初始化进程组
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()

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
    for data, target in dataloader:
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
