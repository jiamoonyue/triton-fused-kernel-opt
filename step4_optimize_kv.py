"""KV Cache 优化：压力测试 + 量化方案
1. 测不同 seq_len 下最大 batch size
2. 计算加入 INT8 KV Cache 量化后的提升
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "D:/LeetCUDA-main/triton_learn/model"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

config = model.config
dtype_size = 2
n_layers = config.num_hidden_layers
n_kv_heads = config.num_key_value_heads
d_head = config.hidden_size // config.num_attention_heads

def kv_cache_mem(seq_len, batch, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes

# 从 1 开始加 batch，直到 OOM
print("=== KV Cache 压力测试 ===")
print(f"模型显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"可用显存: {(6.0 - torch.cuda.memory_allocated()/1024**3):.2f} GB (近似)")
print()

for seq_len in [128, 256, 512, 1024, 2048, 4096]:
    max_batch = 1
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        kv = kv_cache_mem(seq_len, bs) / 1024**3
        model_mem = 3.0
        total = model_mem + kv
        if total < 5.5:  # 留一点余量
            max_batch = bs
        else:
            break

    kv_full = kv_cache_mem(seq_len, max_batch) / 1024**3
    kv_int8 = kv / 2  # INT8 量化后
    total_int8 = 3.0 + kv_int8
    # INT8 能多支持的 batch
    max_batch_int8 = 1
    for bs in range(max_batch + 1, 512):
        kv_test = kv_cache_mem(seq_len, bs, dtype_bytes=1) / 1024**3
        if 3.0 + kv_test < 5.5:
            max_batch_int8 = bs
        else:
            break

    print(f"seq_len={seq_len:4d}: max batch={max_batch:3d} | "
          f"KV Cache={kv_full:.1f} GB | "
          f"INT8后可到 batch={max_batch_int8:3d} ({(max_batch_int8/max_batch - 1)*100:.0f}% 提升)")

print()
print("=== 结论 ===")
print(f"""
KV Cache 是 LLM 推理的显存瓶颈。
每个 token 的 KV Cache ≈ {kv_cache_mem(1,1)/1024:.0f} KB（{n_layers}层×{n_kv_heads}KV头×{d_head}维×2=K+V×2字节）。

优化手段                          效果
───────────────────────────────────────
GQA（{config.num_attention_heads}→{n_kv_heads} KV头）       KV Cache 缩小 {(config.num_attention_heads/n_kv_heads):.0f}x（已应用）
KV Cache INT8 量化                  KV Cache 缩小 2x
KV Cache INT4 量化                  KV Cache 缩小 4x
PagedAttention（vLLM）              消除碎片，利用率 60%→95%
""")
