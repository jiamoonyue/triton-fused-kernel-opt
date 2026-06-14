"""Nsight 分析：优化后 Triton RMSNorm (1 kernel)"""
import torch, triton, triton.language as tl
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0); offs = tl.arange(0, BLOCK_SIZE); mask = offs < n_cols
    x = tl.load(x_ptr+row_idx*n_cols+offs, mask=mask, other=0.0).to(tl.float32)
    x_sq = x*x; mean_sq = tl.sum(x_sq,axis=0)/n_cols; rms = tl.sqrt(mean_sq+eps)
    x_norm = x/rms; w = tl.load(w_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm*w; tl.store(y_ptr+row_idx*n_cols+offs, y.to(tl.float16), mask=mask)
H=1536; w=torch.randn(H,device='cuda',dtype=torch.float16)
x=torch.randn(1024,H,device='cuda',dtype=torch.float16); y=torch.empty_like(x)
for _ in range(10): rmsnorm_kernel[(1024,)](x,w,y,1024,H,1e-6,BLOCK_SIZE=2048)
torch.cuda.synchronize()
rmsnorm_kernel[(1024,)](x,w,y,1024,H,1e-6,BLOCK_SIZE=2048)
torch.cuda.synchronize()
print("Done.")
