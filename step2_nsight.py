"""第2步：模型级 Nsight 分析"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "D:/LeetCUDA-main/triton_learn/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda",
)

inputs = tokenizer("写一首诗：", return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**inputs, use_cache=True)

torch.cuda.synchronize()
print("Done.")
