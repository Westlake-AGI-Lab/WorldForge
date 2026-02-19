#!/bin/bash
# WorldForge (LongCat) Video Generation - Batch Inference Script
# Usage: bash longcat_for_worldforge/run_test_case.sh  (run from project root)
# export HF_ENDPOINT=https://hf-mirror.com  # Uncomment if you need a HuggingFace mirror
cd "$(dirname "$0")"
# ==================== Configuration ====================
CHECKPOINT_DIR="/path/to/LongCat-Video"        # Model weights directory
VIDEO_REF="../test_case/truck/imgs"            # Input frames + masks directory

OUTPUT_DIR="./output"                  # Output directory
SCENE="truck"                                # Scene name (see prompts.py)
NUM_FRAMES=49                                 # Number of output frames
RESOLUTION="480p"                             # 480p or 720p
STATIC="True"                                 # True for static scenes
FPS=16

USE_DISTILL=false                             # true=16-step distill, false=50-step standard
UPSCALE=true                                 # true=upscale to 720p
NUM_INFERENCE_STEPS=$( [ "$USE_DISTILL" = true ] && echo 16 || echo 50 )

# ==================== Parameter Grid ====================
# Modify these arrays to sweep different parameter combinations
omegas=(4)                    # DSG strength (recommended: (4 6))
guidance_scales=(4)           # CFG scale (recommended: (4);ignored if USE_DISTILL=true)
max_channels=(2)              # Max FLF replacement channels per step (recommended: (2 3))
transition_distances=(20)     # Mask softening distance (0=hard edge; recommended: (15 20 25))
step_guide=(28)               # Guide steps (resample guidance applied for first N steps (recommended: (23 25 30 33), for distill mode recommended: (8 9 10 11)))
step_additions=(0)            # Addition to guide_steps for resample_round (recommended: (0 1))

# ==================== Batch Inference ====================
for o in "${omegas[@]}"; do
for g in "${guidance_scales[@]}"; do
for mc in "${max_channels[@]}"; do
for m in "${transition_distances[@]}"; do
for step in "${step_guide[@]}"; do
for add in "${step_additions[@]}"; do
    r_r=$((step + add))
    suffix=$( [ "$USE_DISTILL" = true ] && echo "dt16" || echo "fu50" )
    output="${OUTPUT_DIR}/${suffix}/${RESOLUTION}/o${o}_re2_guide${step}_round${r_r}_mask${m}_max${mc}.mp4"
    
    echo "Running: omega=$o, guide=$step, round=$r_r, mask=$m, max=$mc"
    
    python ./run_longcat_worldforge_single.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --video-ref "$VIDEO_REF" \
        --scene "$SCENE" \
        --seed 42 \
        --resolution "$RESOLUTION" \
        --num-frames $NUM_FRAMES \
        --num-inference-steps $NUM_INFERENCE_STEPS \
        --guidance-scale $g \
        --fps $FPS \
        --guided \
        --resample-steps 2 \
        --guide-steps $step \
        --resample-round $r_r \
        --omega $o \
        --omega_resample $o \
        --soften-mask \
        --transition-distance $m \
        --max-replace $mc \
        --use-pca-channel-selection \
        --static "$STATIC" \
        --output "$output" \
        $( [ "$USE_DISTILL" = true ] && echo "--use_distill" ) \
        $( [ "$UPSCALE" = true ] && echo "--enable-upscale" )
done
done
done
done
done
done
