#!/bin/bash
# DepthCrafter Warping Test Case
# Usage: bash DepthCrafter/run_test_case.sh  (run from project root)

# export HF_ENDPOINT=https://hf-mirror.com  # Uncomment if you need a HuggingFace mirror
cd "$(dirname "$0")"
python warp_depthcrafter.py \
    --video_path ../test_case/case_for_dc/two_car \
    --output_path output_test/two_car \
    --direction up \
    --degree 30 \
    --look_at_depth 0.9 \
    --enable_edge_filter
