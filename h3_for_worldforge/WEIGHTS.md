# Model Weights

## Base Model

Download the **Ref2VA** components from
[MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) and retain this layout:

```text
weights/MiniMax-H3/Ref2VA/
├── transformer/             # model.safetensors.index.json and shards
├── video_vae/source/model.safetensors
├── text_encoder/            # model.safetensors.index.json and shards
└── processor/
```

## Trajectory LoRA

Download the [WorldForge V2 trajectory LoRA](https://huggingface.co/Chenxis27/WorldForge-V2):

```bash
hf download Chenxis27/WorldForge-V2 \
  h3-wfv2.safetensors --local-dir weights
```

[Direct download](https://huggingface.co/Chenxis27/WorldForge-V2/resolve/main/h3-wfv2.safetensors).
The commands use `weights/h3-wfv2.safetensors`; pass another local path with `--lora` if needed.

## 8-Step Acceleration LoRA

Use [Larry v4 step600 EMA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600_ema.safetensors):

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir weights
```

[Direct download](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors).
Pass this file with `--acceleration-lora` for the default 8-step configuration.
The 50-step configuration uses only the base model and trajectory LoRA.
