# RMSNorm 算子融合优化：完整流程

## 项目概述
对 Qwen2.5-1.5B 模型做推理性能分析，通过算子融合实现 RMSNorm 的 3.25x 加速。

## 完整优化流程

### 第1步：Baseline
```
Qwen2.5-1.5B, RTX 4050 6GB, FP16
生成 128 token: 7.1 秒 | 17.9 tokens/s
```

### 第2步：模型级 Nsight 分析（发现瓶颈）
```
Top 最耗时 kernel 全部是 cutlass 矩阵乘:
  matmul: 16.40 ms (77.8%), Memory 94%, Compute 10%

matmul 已是 Tensor Core 极致优化, 手写无法超越。
```

### 第3步：PyTorch Profiler（细化分析）
```
aten::attention          9.69ms (6.8%)
aten::matmul 系列        6.63ms (4.6%)
aten::elementwise (SiLU) 1.50ms (1.0%)
aten::pow                0.39ms (0.3%)  ← RMSNorm 成分
aten::rsqrt              0.25ms (0.2%)  ← RMSNorm 成分
aten::mean               0.13ms (0.1%)  ← RMSNorm 成分

选择 RMSNorm 作为 demo:
  - PyTorch 将其分解为 pow + reduce + mul + rsqrt 4 个基础操作
  - 有算子融合空间, 适合展示 fusion 思想
  - matmul/attention 由 cutlass/cuBLAS 实现, 已达性能上限
```

### 第4步：优化方案
```
将 PyTorch 的 4 个 kernel 融合成 1 个 Triton kernel:
  x→pow→mean→rsqrt→mul→mul→y (4 次显存读写)
  → x→[寄存器全部算完]→y (1 次读 + 1 次写)
```

### 第5步：优化结果
```
优化前 (PyTorch): 0.1300 ms
优化后 (Triton):  0.0400 ms
加速比: 3.25x
```

### 第6步：Nsight 验证
```
优化后 kernel: Memory 82%, Compute 26%, Duration 20us
显存带宽利用充分, 优化方向正确。
```

## 运行方式
```bash
conda activate triton_learn
python benchmark.py                      # 优化前后对比
python step5_int8_kvcache.py             # INT8 KV Cache 量化分析
```

## 环境
- GPU: NVIDIA GeForce RTX 4050 (6GB)
- CUDA 12.4, PyTorch 2.6.0, Triton 3.1.0
- 模型: Qwen2.5-1.5B
