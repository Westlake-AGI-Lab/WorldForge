#!/bin/bash
# WorldForge (WAN) Video Generation - Batch Inference Script
# Usage: bash wan_for_worldforge/run_test_case.sh  (run from project root)
# export HF_ENDPOINT=https://hf-mirror.com  # Uncomment if you need a HuggingFace mirror
cd "$(dirname "$0")"
# ==================== Basic Configuration ====================
MODELS_DIR="/path/to/model_weights"            # Model weights directory (must contain Wan2.1-I2V-14B-480P etc.)
VIDEO_REF="../test_case/truck/imgs"            # Input frames + masks directory
OUTPUT_DIR="./output"                          # Output directory
SCENE="truck"                                   # Scene name (see utils/prompts.py)
NUM_FRAMES=49                                  # Number of output frames
RESOLUTION="480p"                              # 480p or 720p
STATIC="True"                                 # True for static scenes, False for dynamic
NUM_INFERENCE_STEPS=50                         # Diffusion sampling steps

# ==================== Parameter Grid ====================
# Modify these arrays to sweep different parameter combinations
omegas=(4)                  # Auto-guidance strength (recommended: (4 6)) 
guidance_scales=(4)         # CFG scale (recommended: (4) ) 
transition_distances=(15)   # Mask softening distance in pixels (0=hard edge; recommended: (15 20 25))
resample_steps=(2)          # Resampling iterations per step (recommended: (2))
guide_steps=(15 18)         # Guide steps: apply guided fusion for first N steps (recommended: (10 15 18 20 23))
step_additions=(0)        # Extra steps for resample_round = guide_steps + addition (recommended: (0 1))

# ==================== Batch Inference ====================
mkdir -p "$OUTPUT_DIR"

for omega in "${omegas[@]}"; do
for cfg in "${guidance_scales[@]}"; do
for mask in "${transition_distances[@]}"; do
for resample in "${resample_steps[@]}"; do
for guide in "${guide_steps[@]}"; do
for add in "${step_additions[@]}"; do
    round=$((guide + add))
    output="${OUTPUT_DIR}/o${omega}_guide${guide}_round${round}_mask${mask}_cfg${cfg}.mp4"
    
    echo "========================================"
    echo "omega=$omega, guide=$guide, round=$round, mask=$mask, cfg=$cfg"
    echo "output: $output"
    echo "========================================"
    
    python infer_worldforge.py \
        --model "$RESOLUTION" \
        --models-dir "$MODELS_DIR" \
        --video-ref "$VIDEO_REF" \
        --scene "$SCENE" \
        --num-frames $NUM_FRAMES \
        --num-inference-steps $NUM_INFERENCE_STEPS \
        --guidance-scale $cfg \
        --static "$STATIC" \
        --guided \
        --resample-steps $resample \
        --guide-steps $guide \
        --resample-round $round \
        --omega $omega \
        --omega_resample $omega \
        --soften-mask \
        --transition-distance $mask \
        --use-pca-channel-selection \
        --output "$output"
done
done
done
done
done
done
