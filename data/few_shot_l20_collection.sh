#!/usr/bin/env bash
set -euo pipefail

# Frozen experiment: l20_tp4_bf16_qwen3_32b_llama2_13b_two_level_few_shot
mkdir -p data/model_catalog data/profiles/few_shot data/runtime/few_shot data/training/folds data/validation/few_shot

python scripts/audit_model_catalog.py \
  --roots /data/home/public/weight/Qwen3-32B /data/home/public/weight/Llama-2-13b-chat-hf \
  --tp-size 4 --dtype-bytes 2 --accelerator-memory-gb 48 \
  --memory-safety-fraction 0.9 \
  --output-json data/model_catalog/l20_qwen_llama_audit.json \
  --output-csv data/model_catalog/l20_qwen_llama_audit.csv

# Operator profile: Qwen3-32B
python scripts/operator_micro_bench.py \
  --model /data/home/public/weight/Qwen3-32B --stage both --tp-size 4 \
  --batch-sizes 1 \
  --prefill-lengths 128,256,512,1024,2048,3072 \
  --decode-kv-lengths 128,256,512,1024,2048,3072 \
  --warmup 5 --repeats 20 \
  --dtype bfloat16 --platform cuda \
  --output-dir data/profiles/few_shot/qwen3_32b

# Operator profile: Llama-2-13b-chat-hf
python scripts/operator_micro_bench.py \
  --model /data/home/public/weight/Llama-2-13b-chat-hf --stage both --tp-size 4 \
  --batch-sizes 1 \
  --prefill-lengths 128,256,512,1024,2048,3072 \
  --decode-kv-lengths 128,256,512,1024,2048,3072 \
  --warmup 5 --repeats 20 \
  --dtype bfloat16 --platform cuda \
  --output-dir data/profiles/few_shot/llama_2_13b_chat_hf

# TP4 collective profile (run on four GPUs)
torchrun --standalone --nproc_per_node=4 scripts/collective_micro_bench.py \
  --dtype bfloat16 --platform cuda --warmup 10 --repeats 50 \
  --output-dir data/profiles/few_shot/collective_tp4

# Strict offline vLLM truth: Qwen3-32B
python scripts/collect_fixed_batch_suite.py \
  --collector-mode offline --offline-engine-lifecycle sample \
  --model /data/home/public/weight/Qwen3-32B --tokenizer /data/home/public/weight/Qwen3-32B \
  --tp-size 4 --dtype bfloat16 \
  --exact-case prefill:1:128 \
  --exact-case prefill:1:256 \
  --exact-case prefill:1:512 \
  --exact-case prefill:1:1024 \
  --exact-case prefill:1:2048 \
  --exact-case prefill:1:3072 \
  --exact-case decode:1:128 \
  --exact-case decode:1:256 \
  --exact-case decode:1:512 \
  --exact-case decode:1:1024 \
  --exact-case decode:1:2048 \
  --exact-case decode:1:3072 \
  --decode-tokens 32 \
  --warmup 2 --repeats 5 \
  --deployment-config configs/experiments/l20_qwen3_tp4_bf16.json \
  --dataset-role benchmark_on_grid_validation --resume \
  --output-dir data/runtime/few_shot/qwen3_32b

# Strict offline vLLM truth: Llama-2-13b-chat-hf
python scripts/collect_fixed_batch_suite.py \
  --collector-mode offline --offline-engine-lifecycle sample \
  --model /data/home/public/weight/Llama-2-13b-chat-hf --tokenizer /data/home/public/weight/Llama-2-13b-chat-hf \
  --tp-size 4 --dtype bfloat16 \
  --exact-case prefill:1:128 \
  --exact-case prefill:1:256 \
  --exact-case prefill:1:512 \
  --exact-case prefill:1:1024 \
  --exact-case prefill:1:2048 \
  --exact-case prefill:1:3072 \
  --exact-case decode:1:128 \
  --exact-case decode:1:256 \
  --exact-case decode:1:512 \
  --exact-case decode:1:1024 \
  --exact-case decode:1:2048 \
  --exact-case decode:1:3072 \
  --decode-tokens 32 \
  --warmup 2 --repeats 5 \
  --deployment-config configs/experiments/l20_llama2_tp4_bf16.json \
  --dataset-role benchmark_on_grid_validation --resume \
  --output-dir data/runtime/few_shot/llama_2_13b_chat_hf
