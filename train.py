"""LoRA SFT of LFM3-1.7B on transaction categorization (per docs.liquid.ai TRL guide)."""
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "LiquidAI/LFM3-1.7B"
OUT = "lfm3-txn-lora"
max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # smoke test: `python train.py 2`

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=load_dataset("json", data_files="data/train.jsonl", split="train"),
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    args=SFTConfig(
        output_dir=OUT,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=torch.cuda.is_available(),
        max_length=256,
        assistant_only_loss=True,  # chat template marks assistant turns with {% generation %}
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    ),
)
trainer.train()
trainer.save_model(OUT)
