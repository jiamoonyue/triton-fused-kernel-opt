"""Nsight 分析：优化前 PyTorch RMSNorm (4 kernel)"""
import torch
def torch_rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * weight
H = 1536
weight = torch.randn(H, device='cuda', dtype=torch.float16)
x = torch.randn(1024, H, device='cuda', dtype=torch.float16)
for _ in range(10): _ = torch_rmsnorm(x, weight)
torch.cuda.synchronize()
_ = torch_rmsnorm(x, weight)
torch.cuda.synchronize()
print("Done.")
