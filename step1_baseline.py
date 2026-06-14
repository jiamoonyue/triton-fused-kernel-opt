"""第1步：跑模型 baseline"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "D:/LeetCUDA-main/triton_learn/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda",
)

print(f"参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
print(f"显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")

prompt = "写一首关于春天的诗："
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

for _ in range(5):
    _ = model.generate(**inputs, max_new_tokens=64)
torch.cuda.synchronize()

n = 10
torch.cuda.synchronize()
start = time.time()
for _ in range(n):
    _ = model.generate(**inputs, max_new_tokens=128)
torch.cuda.synchronize()
end = time.time()

avg = (end - start) / n
print(f"生成128 token平均: {avg:.2f}秒 | {128/avg:.1f} tokens/s")
