include .env
export HF_TOKEN
export HF_HOME := $(CURDIR)/.hf-cache
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
PY := .venv/bin/python
DATASET_URL := https://huggingface.co/datasets/mitulshah/transaction-categorization/resolve/main/default/train/0000.parquet

GGUF := lfm3-txn-q8_0.gguf
LLAMA_BIN := llama.cpp/build/bin

.PHONY: install train eval eval-base smoke all merge gguf push eval-gguf

all: train eval eval-base

install:
	uv sync --python /usr/bin/python3

data/train.parquet:
	mkdir -p data
	curl -fL -H "Authorization: Bearer $$HF_TOKEN" -o $@ $(DATASET_URL)

data/train.jsonl data/eval.jsonl &: data/train.parquet prepare.py
	$(PY) prepare.py

train: data/train.jsonl
	$(PY) train.py

eval: data/eval.jsonl
	$(PY) evaluate.py lfm3-txn-lora

eval-base: data/eval.jsonl
	$(PY) evaluate.py base

smoke: data/train.jsonl
	$(PY) train.py 2
	$(PY) evaluate.py lfm3-txn-lora 64

llama.cpp:
	git clone --depth 1 https://github.com/ggml-org/llama.cpp

lfm3-txn-merged/model.safetensors: lfm3-txn-lora/adapter_model.safetensors merge.py
	$(PY) merge.py

merge: lfm3-txn-merged/model.safetensors

$(GGUF): lfm3-txn-merged/model.safetensors | llama.cpp
	$(PY) llama.cpp/convert_hf_to_gguf.py lfm3-txn-merged --outtype q8_0 --outfile $@

gguf: $(GGUF)

$(LLAMA_BIN)/llama-server $(LLAMA_BIN)/llama-cli &: | llama.cpp
	cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
	cmake --build llama.cpp/build -j --target llama-server llama-cli

eval-gguf: $(GGUF) $(LLAMA_BIN)/llama-server data/eval.jsonl
	$(PY) eval_gguf.py $(LLAMA_BIN)/llama-server $(GGUF)

push: $(GGUF)
	$(PY) push.py
