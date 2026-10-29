"""Exact-match accuracy on data/eval.jsonl. Usage: python evaluate.py [adapter_dir] [limit]"""
import json
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM3-1.7B"
adapter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "base" else None
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(device)
if adapter:
    model.load_adapter(adapter)
model.eval()

rows = [json.loads(line)["messages"] for line in open("data/eval.jsonl")][:limit]
correct, errors = 0, Counter()
for i in range(0, len(rows), 64):
    batch = rows[i:i + 64]
    prompts = [tokenizer.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for m in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
    for m, ids in zip(batch, out[:, enc.input_ids.shape[1]:]):
        pred, gold = tokenizer.decode(ids, skip_special_tokens=True).strip(), m[-1]["content"]
        correct += pred == gold
        if pred != gold:
            errors[(gold, pred)] += 1

print(f"accuracy {correct / len(rows):.4f} ({correct}/{len(rows)})")
for (gold, pred), n in errors.most_common(10):
    print(f"{n:4d}  {gold!r} -> {pred!r}")
