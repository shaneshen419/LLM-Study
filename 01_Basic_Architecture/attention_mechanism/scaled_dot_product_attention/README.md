# Scaled Dot Product Attention（SDPA）

在学习huggingFace的Transformer库时，我们不可避免会遇到scaled_dot_product_attention(SDPA)这个函数，它被用来加速大模型的Attention计算，本文就详细介绍一下它的使用方法，核心内容主要参考了torch.nn.functional中该函数的注释。

## 一、Attention计算公式

Attention的计算主要涉及三个矩阵：Q、K、V。我们先不考虑multi-head attention，只考虑one head的self attention。在大模型的prefill阶段，这三个矩阵的维度均为N x d，N即为上下文的长度；在decode阶段，Q的维度为1 x d， KV还是N x d。然后通过下面的公式计算attention矩阵：

$$
O=\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

在真正使用attention的时候，我们往往采用multi-head attention(MHA)。MHA的计算公式和one head attention基本一致，它改变了Q、K、V每一行的定义：将维度d的向量分成h组变成一个h x dk的矩阵，Q、K、V此时成为了 $N \cdot h \cdot d_k$ 的三维矩阵（不考虑batch维）。分别将Q、K、V的第一和第二维进行转置得到三个维度为 $h \cdot N \cdot d_k$ 的三维矩阵。此时的三个矩阵就是具有h个头的Q、K、V，我们就可以按照self attention的定义计算h个头的attention值。

不过，在真正进行大模型推理的时候就会发现KV Cache是非常占显存的，所以大家尝试各种手段压缩KV Cache，具体可以参考[《大模型推理--KV Cache压缩》](https://zhuanlan.zhihu.com/p/691038809)。一种手段就是将MHA替换成group query attention(GQA)，这块在torch2.5以上的SDPA中也已经得到了支持。

## [二、SDPA伪代码](./sdpa.py)

可以看出，我们实际在使用SDPA时除了query、key和value之外，还有另外几个参数：attn_mask、dropout_p、is_causal、scale和enable_gqa。scale就是计算Attention时的缩放因子，一般无需传递。dropout_p表示Dropout概率，在推理阶段也不需要传递，不过官方建议如下输入：dropout_p=(self.p if self.training else 0.0)。我们着重看一下另外三个参数在使用时该如何设置。

先看enable_gqa。前面提到GQA是一种KV Cache压缩方法，MHA的KV和Q一样，也会有h个头，GQA则将KV的h个头进行压缩来减小KV Cache的大小。比如Qwen2-7B-Instruct这个模型，Q的h等于28，KV的h等于4，相当于把KV Cache压缩到之前的七分之一。GQA虽然压缩了KV Cache，但是真正要计算Attention的时候还是需要对齐KV与Q的head数，所以我们可以看到HF Transformer库中的qwen2.py在Attention计算时会有一个repeat_kv的操作，目的就是将QKV的head数统一。在torch2.5以后的版本中，我们无需再手动去执行repeat_kv，直接将SDPA的enable_gqa设置为True即可自动完成repeat_kv，而且速度比自己去做repaet_kv还要更快。

attn_mask和is_causal两个参数的作用相同，目的都是要给softmax之前的QKT矩阵添加mask。只不过attn_mask是自己在外面构造mask矩阵，is_causal则是根据大模型推理的阶段属于prefill还是decode来进行设置。通过看伪代码可以看出，SDPA会首先构造一个L x S的零矩阵attn_bias，L表示Q的上下文长度，S表示KV Cache的长度。在prefill阶段，L和S相等，在decode阶段，L为1，S还是N。所以在prefill阶段，attn_bias就是一个N x N的矩阵，将is_causal设置为True时就会构造一个下三角为0，上三角为负无穷的矩阵作为attn_bias，然后将其加到QKT矩阵上，这样就实现了因果关系的Attention计算。在decode阶段，attn_bias就是一个1 x N的向量，此时可以将is_causal设置为False，attn_bias始终为0就不会对 $QK^T$ 行向量产生影响，表示KV Cache所有的行都参与计算，因果关系保持正确。

attn_mask作用和is_causal一样，但是需要我们自行构造，如果你对如何构造不了解建议就使用is_causal选项，prefill阶段设置为True，decode阶段设置为False，attn_mask设置为None。不过，如果prefill按照chunk来执行也即chunk_prefill阶段，我们会发现is_causal设置为True时的attn_bias设置的不正确，我们不是从左上角开始构造下三角矩阵，而是要从右下角开始构造下三角矩阵，这种情况下我们可以从外面自行构造attn_mask矩阵代替SDPA的构造。attn_mask有两种构造方式，一种是bool类型，True的位置会保持不变，False的位置会置为负无穷；一种是float类型，会直接将attn_mask加到SDPA内部的attn_bias上，和bool类型一样，我们一般是构造一个下三角为0上三角为负无穷的矩阵。总结来说，绝大多数情况下我们只需要设置is_causal选项，prefill阶段设置为True，decode阶段设置为False，attn_mask设置为None即可。如果推理阶段引入了chunk_prefill，则我们需要自行构造attn_mask，但是要注意构造的attn_mask矩阵是从右下角开始的下三角矩阵。

## 三、SDPA实现(翻译自SDPA注释)

目前SDPA有三种实现：

 - 基于FlashAttention-2的实现；
 - Memory-Efficient Attention(facebook xformers)；
 - Pytorch版本对上述伪代码的c++实现(对应MATH后端)。

针对CUDA后端，SDPA可能会调用经过优化的内核以提高性能。对于所有其他后端，将使用PyTorch实现。所有实现方式默认都是启用的，SDPA会尝试根据输入自动选择最优的实现方式。为了对使用哪种实现方式提供更细粒度的控制，torch提供了以下函数来启用和禁用各种实现方式：

 - torch.nn.attention.sdpa_kernel：一个上下文管理器，用于启用或禁用任何一种实现方式；
 - torch.backends.cuda.enable_flash_sdp：全局启用或禁用FlashAttention；
 - torch.backends.cuda.enable_mem_efficient_sdp：全局启用或禁用memory efficient attention；
 - torch.backends.cuda.enable_math_sdp：全局启用或禁用PyTorch的C++实现。

每个融合内核都有特定的输入限制。如果用户需要使用特定的融合实现方式，请使用torch.nn.attention.sdpa_kernel禁用PyTorch的C++实现。如果某个融合实现方式不可用，将会发出警告，说明该融合实现方式无法运行的原因。由于融合浮点运算的特性，此函数的输出可能会因所选择的后端内核而异。C++实现支持torch.float64，当需要更高精度时可以使用。对于math后端，如果输入是torch.half或torch.bfloat16类型，那么所有中间计算结果都会保持为torch.float类型。

