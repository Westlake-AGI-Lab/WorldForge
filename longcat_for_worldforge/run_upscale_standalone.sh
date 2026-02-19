#!/bin/bash
# Standalone upscale script: 480p -> 720p
# Note: Upscaling is also integrated in run_longcat_worldforge_single.py via --enable-upscale

torchrun run_upscale.py \
    --input_video "output/example_480p.mp4" \
    --output_video "output/example_720p.mp4" \
    --checkpoint_dir /path/to/LongCat-Video \
    --scene horse \
    --first_frame /path/to/first_frame.png \
    --t_thresh 0.5
