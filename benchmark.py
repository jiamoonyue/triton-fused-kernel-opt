"""Fused RMSNorm benchmark: PyTorch vs Triton
优化前 4 kernel → 优化后 1 kernel, 3.25x 加速
"""
import torch, triton, triton.language as tl, time

def torch_rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * weight

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row_idx * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)
    x_norm = x / rms
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w
    tl.store(y_ptr + row_idx * n_cols + offs, y.to(tl.float16), mask=mask)

def triton_rmsnorm(x, weight):
    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    rmsnorm_kernel[grid](x, weight, y, n_rows, n_cols, 1e-6, BLOCK_SIZE=2048)
    return y

H = 1536
weight = torch.randn(H, device='cuda', dtype=torch.float16)
x = torch.randn(1024, H, device='cuda', dtype=torch.float16)

for _ in range(10):
    _ = torch_rmsnorm(x, weight); _ = triton_rmsnorm(x, weight)
torch.cuda.synchronize()

for name, fn in [("PyTorch RMSNorm", lambda: torch_rmsnorm(x, weight)),
                 ("Triton RMSNorm",  lambda: triton_rmsnorm(x, weight))]:
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(100): fn()
    torch.cuda.synchronize()
    avg = (time.time()-s)/100*1000
    print(f"{name:>20}: {avg:.4f} ms")
