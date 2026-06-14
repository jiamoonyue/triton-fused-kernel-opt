"""KV Cache INT8 量化演示
核心思路：K 和 V 存 INT8（省一半），Attention 时解量化为 FP16
"""
import torch
import time

# 模拟 KV Cache 的维度
batch_size = 1
n_layers = 28
n_kv_heads = 2
d_head = 128
seq_len = 4096
shape = (batch_size, n_kv_heads, seq_len, d_head)

print(f"KV Cache 维度: {shape}")
print(f"总元素数: {batch_size * n_kv_heads * seq_len * d_head * n_layers * 2:,} (每层 K + V)")
print()

# FP16 KV Cache（当前做法）
k_fp16 = torch.randn(shape, device='cuda', dtype=torch.float16)
v_fp16 = torch.randn(shape, device='cuda', dtype=torch.float16)

# 存一次的显存
fp16_bytes_per_token = k_fp16.numel() * 2 + v_fp16.numel() * 2
print(f"每个 token 的 KV Cache: {fp16_bytes_per_token / 1024:.0f} KB (FP16)")
print(f"seq_len={seq_len}: {fp16_bytes_per_token * seq_len / 1024**2:.0f} MB (FP16)")
print()

# ========== INT8 量化方案 ==========
# 对每个 head 单独做 per-group 量化
# scale = max(|values|) / 127

k_int8 = k_fp16.to(torch.int8)
v_int8 = v_fp16.to(torch.int8)

# 简化版：per-tensor 量化
k_scale = k_fp16.abs().max() / 127
v_scale = v_fp16.abs().max() / 127

# 存储：INT8 K/V（半尺寸）+ scale（两个 float16）
int8_bytes = k_int8.numel() * 1 + v_int8.numel() * 1 + 4  # K(INT8) + V(INT8) + 2 scales(FP16)
print(f"INT8 每个 token: {int8_bytes * n_layers / 1024:.0f} KB (含 scale)")
print(f"节省: {(1 - int8_bytes/fp16_bytes_per_token)*100:.1f}%")
print()

# ========== Per-group 量化（更精细）==========
# 每 32 个元素一组，各有自己的 scale
group_size = 32
n_groups = d_head // group_size

k_flat = k_fp16.reshape(-1, d_head)
k_groups = k_flat.reshape(-1, n_groups, group_size)
k_gmax = k_groups.abs().max(dim=-1).values / 127
k_gquant = (k_groups / k_gmax.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

# Per-group 的存储
group_bytes = (k_int8.numel() * 1  # INT8 K
               + k_gmax.numel() * 2)  # FP16 scales
print(f"Per-group INT8 (group={group_size}):")
print(f"  数据: {k_int8.numel() * 1 / 1024:.0f} KB (INT8)")
print(f"  参数: {k_gmax.numel() * 2 / 1024:.0f} KB (scales)")
print(f"  总计: {group_bytes / 1024:.0f} KB")
print(f"  节省: {(1 - group_bytes/fp16_bytes_per_token)*100:.1f}%")
print()

# ========== 解量化验证 ==========
# 从 INT8 恢复到 FP16
k_restored = k_int8.float() * k_scale / 127
error = (k_restored - k_fp16.half()).abs().mean().item()
print(f"Per-tensor 解量化误差: {error:.6f}")

k_restored_group = k_gquant.float() * k_gmax.unsqueeze(-1)
k_groups_fp16 = k_groups.half()
error_group = (k_restored_group - k_groups_fp16).abs().mean().item()
print(f"Per-group 解量化误差: {error_group:.6f}")

# ========== Attention 中怎么用 ==========
# 伪代码：Attention 需要 Q x K^T
# K_INT8 → 解量化 → K_FP16 → Q @ K_FP16^T
# 或者在 INT8 域直接算（需要 INT8 matmul 硬件支持）

# ========== 性能基准 ==========
print()
print("=" * 50)
print("量化/解量化性能")
print("=" * 50)

n_warmup, n_iter = 100, 1000

def bench(fn, name):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    avg_us = (time.time() - s) / n_iter * 1e6
    print(f"  {name:>35}: {avg_us:8.1f} us")
    return avg_us

# 量化
bench(lambda: k_fp16.to(torch.int8), "量化 (FP16→INT8)")
# 解量化
scale = k_fp16.abs().max() / 127
k_restore_tensor = k_int8.float() * scale
bench(lambda: k_int8.float() * scale / 127, "解量化 (INT8→FP16)")

# 对比：FP16 matmul vs INT8 后的 FP16 matmul
q = torch.randn(1, n_kv_heads, 1, d_head, device='cuda', dtype=torch.float16)
t_fp16 = bench(lambda: q @ k_fp16.transpose(-2, -1), "Attention: Q@K^T (FP16)")
t_int8_deq = bench(
    lambda: q @ (k_int8.float().to(torch.float16)).transpose(-2, -1),
    "Attention: Q@K^T (INT8→FP16)"
)

print(f"\n结论：")
print(f"  FP16 Attention 时间: {t_fp16:.1f} us")
print(f"  INT8 解量化后: {t_int8_deq:.1f} us (多了解量化开销)")
print(f"  说明：INT8 省显存，但解量化有额外开销")
print(f"  实际部署用 INT8 matmul 硬件（Tensor Core INT8）避免解量化")
