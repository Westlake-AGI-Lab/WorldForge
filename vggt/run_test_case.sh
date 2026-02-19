#!/bin/bash
# VGGT Warping Test Case
# Usage: bash vggt/run_test_case.sh  (run from project root)

# export HF_ENDPOINT=https://hf-mirror.com  # Uncomment if you need a HuggingFace mirror
cd "$(dirname "$0")"
python run_warp.py \
    --image_path ../test_case/case_for_vggt/truck \
    --output_path output_test \
    --camera 2 \
    --direction left \
    --degree 20 \
    --frame_single 25 \
    --look_at_depth 0.25
