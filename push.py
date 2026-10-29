"""Upload the adapter, GGUF, model card and license to the Hugging Face Hub in one commit."""
from huggingface_hub import CommitOperationAdd, HfApi

REPO = "oleggat0r/LFM3-1.7B-transaction-categorization"
FILES = {
    "README.md": "hf-upload/README.md",
    "LICENSE": "hf-upload/LICENSE",
    "adapter_config.json": "lfm3-txn-lora/adapter_config.json",
    "adapter_model.safetensors": "lfm3-txn-lora/adapter_model.safetensors",
    "tokenizer.json": "lfm3-txn-lora/tokenizer.json",
    "tokenizer_config.json": "lfm3-txn-lora/tokenizer_config.json",
    "chat_template.jinja": "lfm3-txn-lora/chat_template.jinja",
    "lfm3-txn-q8_0.gguf": "lfm3-txn-q8_0.gguf",
}

api = HfApi()
api.create_repo(REPO, exist_ok=True)
info = api.create_commit(REPO, operations=[CommitOperationAdd(k, v) for k, v in FILES.items()],
                         commit_message="Upload LoRA adapter and Q8_0 GGUF")
print(info.commit_url)
