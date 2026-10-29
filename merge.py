"""Merge the LoRA adapter into the base weights for GGUF conversion."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM3-1.7B"
ADAPTER = "lfm3-txn-lora"
OUT = "lfm3-txn-merged"

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.save_pretrained(OUT)
AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True).save_pretrained(OUT)
