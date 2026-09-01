#!/usr/bin/env bash
# End-to-end plumbing check with no GPU and no model download beyond a 0.5B.
# Uses the mock backend, so it exercises every join and index but no real model.
set -euo pipefail
RUN=runs/smoke
rm -rf "$RUN"
python scripts/01_build_dataset.py --dataset gsm8k --n 20 --out "$RUN"
python scripts/02_generate.py --run "$RUN" --backend mock --model Qwen/Qwen2.5-7B-Instruct
python scripts/03_execute.py --run "$RUN"
echo "smoke ok"

# Look at one problem in full: prompt, completion, answer recovery, token alignment.
python scripts/inspect_one.py --run "$RUN" --index 0 --model Qwen/Qwen2.5-7B-Instruct
