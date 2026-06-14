"""提取 Nsight 结果中的 Top 20 最耗时 kernel"""
import subprocess

ncu = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"
result = subprocess.run(
    [ncu, "--import", "step2_model.ncu-rep", "--page", "details"],
    capture_output=True, text=True, timeout=120, shell=True,
)

lines = result.stdout.split('\n')

# 找到所有 kernel entry（带有 Context, Device, CC 的行）
kernel_entries = []
for i, line in enumerate(lines):
    if (", Context" in line and "Device" in line and "CC" in line
            and not line.startswith("==")):
        name = line.strip()
        dur = 0.0
        mem = 0.0
        cmpval = 0.0
        # 往下找 Duration, Memory, Compute
        for j in range(i + 1, min(i + 20, len(lines))):
            lj = lines[j].strip()
            if lj.startswith("Duration") and "usecond" in lj:
                try:
                    dur = float(lj.split()[-1])
                except: pass
            elif lj.startswith("Memory Throughput") and "%" in lj:
                try:
                    mem = float(lj.split()[-1])
                except: pass
            elif lj.startswith("Compute (SM) Throughput") and "%" in lj:
                try:
                    cmpval = float(lj.split()[-1])
                except: pass
            if dur > 0 and mem > 0 and cmpval > 0:
                break

        kernel_entries.append((name, dur, mem, cmpval))

# 按 Duration 排序
kernel_entries.sort(key=lambda x: x[1], reverse=True)

print(f"{'Kernel':<75} {'Dur(us)':>10} {'Mem%':>8} {'Cmp%':>8}")
print("=" * 101)
for name, dur, mem, cmpval in kernel_entries[:20]:
    short = name.split("(")[0].strip()
    if len(short) > 73:
        short = short[:70] + "..."
    print(f"{short:<75} {dur:>10.1f} {mem:>7.1f}% {cmpval:>7.1f}%")

# 按类型归类
print("\n=== 按类型分组（总 Duration）===")
cats = {"embedding": 0, "matmul": 0, "elementwise": 0, "reduce": 0,
        "attention": 0, "other": 0}
for name, dur, _, _ in kernel_entries:
    nl = name.lower()
    if "embed" in nl:
        cats["embedding"] += dur
    elif any(k in nl for k in ["mm_kernel", "addmm", "linear", "gemv", "gemm"]):
        cats["matmul"] += dur
    elif "reduce" in nl:
        cats["reduce"] += dur
    elif "attention" in nl or "sdpa" in nl:
        cats["attention"] += dur
    elif "elementwise" in nl or "vectorized" in nl or "pow" in nl or "rsqrt" in nl:
        cats["elementwise"] += dur
    else:
        cats["other"] += dur

total = sum(cats.values())
if total > 0:
    for cat, t in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:15s}: {t/1000:8.3f} ms | {t/total*100:5.1f}%")
