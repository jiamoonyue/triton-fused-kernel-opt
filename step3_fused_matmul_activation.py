"""融合优化：gate_proj + up_proj 双 matmul + SiLU + mul 融合"""
import torch
import triton
import triton.language as tl
import time

@triton.jit
def fused_kernel(
    a_ptr, wg_ptr, wu_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_base = a_ptr + rm[:, None] * stride_am
    wg_base = wg_ptr + rn[None, :] * stride_wn
    wu_base = wu_ptr + rn[None, :] * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kk = k + rk
        a = tl.load(a_base + kk[None, :] * stride_ak, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.)
        wg = tl.load(wg_base + kk[:, None] * stride_wk, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.)
        wu = tl.load(wu_base + kk[:, None] * stride_wk, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.)
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)

    gate = acc_g * tl.sigmoid(acc_g)
    result = gate * acc_u
    c_base = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_base, result.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_fused(x, wg, wu):
    M, K = x.shape
    N = wg.shape[0]
    c = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda m: (triton.cdiv(M, m['BLOCK_M']), triton.cdiv(N, m['BLOCK_N']))
    fused_kernel[grid](x, wg, wu, c, M, N, K,
        x.stride(0), x.stride(1), wg.stride(0), wg.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c


def torch_baseline(x, wg, wu):
    return torch.nn.functional.silu(torch.mm(x, wg.T)) * torch.mm(x, wu.T)


M, N, K = 1, 8960, 1536  # decode 阶段，batch=1
x = torch.randn(M, K, device='cuda', dtype=torch.float16)
wg = torch.randn(N, K, device='cuda', dtype=torch.float16)
wu = torch.randn(N, K, device='cuda', dtype=torch.float16)

ref = torch_baseline(x, wg, wu)
y = triton_fused(x, wg, wu)
diff = (y - ref).abs().max().item()
print(f"误差: {diff:.2f} | 正确: {diff < 100}")
print()

def bench(fn, name):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(200): fn()
    torch.cuda.synchronize()
    a = (time.time()-s)/200*1000
    print(f"{name:>40}: {a:8.4f} ms")
    return a

t0 = bench(lambda: torch_baseline(x, wg, wu), "cutlass: 2 matmuls + SiLU + mul")
t1 = bench(lambda: triton_fused(x, wg, wu), "Triton: 1 fused kernel")
print(f"{'加速比':>40}: {t0/t1:.2f}x")
