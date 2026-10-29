---
base_model: LiquidAI/LFM3-1.7B
datasets:
- mitulshah/transaction-categorization
library_name: peft
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- liquid
- lfm3
- lora
- gguf
- text-classification
- finance
language:
- en
---

# LFM3-1.7B Transaction Categorization

A LoRA fine-tune of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) that maps a bank transaction description to one of 10 categories. It was trained on [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization).

This repository is a modified version (Derivative Work) of LFM3-1.7B. It is distributed under the LFM Open License v1.0; see [LICENSE](LICENSE).

## Files

| File | Description |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | LoRA adapter for `LiquidAI/LFM3-1.7B` |
| `lfm3-txn-q8_0.gguf` | Adapter merged into the base model and converted to Q8_0 GGUF for llama.cpp |

## Results

Exact-match accuracy on 2,000 held-out examples (200 per category). No eval description appears in the training data.

| Model | Accuracy |
|---|---|
| LFM3-1.7B (base) | 5.95% |
| This adapter (bf16) | **99.0%** |

The base model often answers the first listed category or explains its answer instead of giving only the category name. The GGUF was not evaluated separately.

## Prompt format

Use this system prompt exactly, with the transaction description as the user message. Turn thinking off.

```
Classify the bank transaction into exactly one category. Reply with the category name only.
Categories: Charity & Donations; Entertainment & Recreation; Financial Services; Food & Dining; Government & Legal; Healthcare & Medical; Income; Shopping & Retail; Transportation; Utilities & Services
```

### llama.cpp

```sh
llama-cli -m lfm3-txn-q8_0.gguf --reasoning off -n 16 --temp 0 \
  -sys "Classify the bank transaction into exactly one category. Reply with the category name only.
Categories: Charity & Donations; Entertainment & Recreation; Financial Services; Food & Dining; Government & Legal; Healthcare & Medical; Income; Shopping & Retail; Transportation; Utilities & Services" \
  -p "Uber Ride"
```

### Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "LiquidAI/LFM3-1.7B"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "oleggat0r/LFM3-1.7B-transaction-categorization")

SYSTEM = (
    "Classify the bank transaction into exactly one category. Reply with the category name only.\n"
    "Categories: Charity & Donations; Entertainment & Recreation; Financial Services; Food & Dining; "
    "Government & Legal; Healthcare & Medical; Income; Shopping & Retail; Transportation; Utilities & Services"
)
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Uber Ride"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                 return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))  # Transportation
```

## Training

Training code: [oleggator/lfm3-transaction-categorization](https://github.com/oleggator/lfm3-transaction-categorization)

- Data: 10,000 examples (1,000 per category), sampled from the 4.5M-row dataset after dropping duplicates and descriptions with conflicting labels. Input is the description only; country and currency are not used.
- LoRA: r=16, alpha=32, dropout 0.05, all linear layers.
- SFT with TRL: 1 epoch, batch size 16, learning rate 2e-4 with a cosine schedule, loss on assistant tokens only.
- Hardware: one AMD Radeon RX 7900 (ROCm). Training took about 4 minutes.

## Limitations

- The dataset is synthetic and template-based, so accuracy on real bank statements will be lower.
- Some dataset labels are noisy (for example, pharmacy chains labelled Healthcare & Medical, some retail stores labelled Food & Dining). Most remaining errors are of this kind.
- Only the 10 categories above are supported.
