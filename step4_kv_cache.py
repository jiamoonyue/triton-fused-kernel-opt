"""KV Cache 分析：显存占用、GQA 对比、不同 seq_len 的影响"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "D:/LeetCUDA-main/triton_learn/model"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda",
)
config = model.config

n_layers = config.num_hidden_layers      # 28
n_q_heads = config.num_attention_heads   # 12
n_kv_heads = config.num_key_value_heads  # 2 (GQA!)
d_head = config.hidden_size // config.num_attention_heads  # 1536/12 = 128
dtype_size = 2  # FP16 = 2 bytes

print(f"模型配置:")
print(f"  层数: {n_layers}")
print(f"  Q 头数: {n_q_heads}, KV 头数: {n_kv_heads} (GQA, 每组 {n_q_heads//n_kv_heads} 个Q头共享1组KV)")
print(f"  每头维度: {d_head}")
print(f"  精度: FP16 ({dtype_size} bytes)")

def kv_cache_size(n_layers, n_kv_heads, d_head, seq_len, batch_size, dtype_size):
    """KV Cache 显存 = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype"""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * dtype_size

def kv_cache_mha_size(n_layers, n_q_heads, d_head, seq_len, batch_size, dtype_size):
    """如果是 MHA（不使用 GQA），KV Cache 会多大"""
    return 2 * n_layers * n_q_heads * d_head * seq_len * batch_size * dtype_size

print(f"\n{'='*60}")
print(f"KV Cache 显存占用分析")
print(f"{'='*60}")

for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
    for bs in [1, 4, 16]:
        kv = kv_cache_size(n_layers, n_kv_heads, d_head, seq_len, bs, dtype_size)
        kv_mha = kv_cache_mha_size(n_layers, n_q_heads, d_head, seq_len, bs, dtype_size)
        kv_mb = kv / (1024**2)
        kv_mha_mb = kv_mha / (1024**2)
        save_pct = (1 - kv/kv_mha) * 100
        if seq_len == 128 and bs == 1:
            print(f"seq={seq_len:4d} batch={bs:2d} | GQA: {kv_mb:8.1f} MB | MHA: {kv_mha_mb:8.1f} MB | GQA 节省 {save_pct:.0f}%")
        elif bs == 1:
            print(f"seq={seq_len:4d} batch={bs:2d} | GQA: {kv_mb:8.1f} MB | MHA: {kv_mha_mb:8.1f} MB")

# 实际跑一下，看显存怎么涨
print(f"\n{'='*60}")
print(f"实际推理显存变化")
print(f"{'='*60}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

for max_tokens in [32, 128, 512]:
    inputs = tokenizer("写一首诗：", return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    _ = model.generate(**inputs, max_new_tokens=max_tokens)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"max_new_tokens={max_tokens:3d}: 峰值显存 {peak:.2f} GB")

print(f"\n{'='*60}")
print(f"结论")
print(f"{'='*60}")
print(f"""
GQA (12 Q头 / 2 KV头) 相比 MHA:
  每个 token 的 KV Cache = {kv_cache_size(n_layers,n_kv_heads,d_head,1,1,dtype_size)/1024:.0f} KB/token

如果不用 GQA（MHA）：
  每个 token 的 KV Cache = {kv_cache_mha_size(n_layers,n_q_heads,d_head,1,1,dtype_size)/1024:.0f} KB/token

在 seq_len=4096 时：
  GQA KV Cache = {kv_cache_size(n_layers,n_kv_heads,d_head,4096,1,dtype_size)/1024**2:.0f} MB
  MHA KV Cache = {kv_cache_mha_size(n_layers,n_q_heads,d_head,4096,1,dtype_size)/1024**2:.0f} MB
  → GQA 节省 {(1-kv_cache_size(n_layers,n_kv_heads,d_head,4096,1,dtype_size)/kv_cache_mha_size(n_layers,n_q_heads,d_head,4096,1,dtype_size))*100:.0f}%
""")
