#!/usr/bin/env bash
# Full run on the GPU box. Steps 2 and 4 need CUDA; 1, 3 and 5 do not.
#
#   ./scripts/run_remote.sh runs/gsm8k_1k gsm8k 1000
set -euo pipefail

RUN="${1:?usage: run_remote.sh <run_dir> <dataset> <n>}"
DATASET="${2:-gsm8k}"
N="${3:-1000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== 1/5 build dataset ==="
python scripts/01_build_dataset.py --dataset "$DATASET" --n "$N" --out "$RUN"

echo "=== 2/5 generate (vLLM) ==="
python scripts/02_generate.py --run "$RUN" --backend vllm --model "$MODEL" \
    --temperature 0.0 --max-tokens 1024

echo "=== 3/5 execute PoT programs ==="
python scripts/03_execute.py --run "$RUN" --workers 8

echo "=== 4/5 harvest activations (HF) ==="
# Separate process: vLLM holds the GPU for the whole of step 2, so the harvest
# must not share it. Run this only after step 2 has fully exited.
python scripts/04_harvest_activations.py --run "$RUN" --model "$MODEL" \
    --device cuda --dtype float16 --layer-stride 1

echo "=== 5/5 probes ==="
python scripts/05_train_probes.py --run "$RUN" --scheme first_token --n-permutations 50

echo
echo "done. results: $RUN/summary/"
