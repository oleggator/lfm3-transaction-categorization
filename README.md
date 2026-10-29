# lfm3-transaction-categorization

LoRA fine-tune of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization). It reaches 99.0% accuracy on 2,000 held-out examples; the base model scores 5.95%.

Model: [oleggat0r/LFM3-1.7B-transaction-categorization](https://huggingface.co/oleggat0r/LFM3-1.7B-transaction-categorization)

## Usage

Requires [uv](https://docs.astral.sh/uv/), an AMD GPU with ROCm (torch is pinned to ROCm wheels), and `HF_TOKEN=...` in `.env`.

```sh
make install   # dependencies
make           # download data, train, evaluate adapter and base model
make gguf      # merge adapter, export Q8_0 GGUF via llama.cpp
make eval-gguf # build llama.cpp, evaluate the GGUF on CPU
make push      # upload adapter and GGUF to the Hugging Face Hub
```
