# External components

This package depends on these separately distributed projects:

| Component | Source | Role |
|---|---|---|
| MiniMax-H3 Ref2VA | [MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) | Transformer, video VAE, text encoder and processor |
| DiffSynth-Studio | [Pinned revision](https://github.com/modelscope/DiffSynth-Studio/tree/c458cb42ab1ee838bff85c6546e14bb01c3571e9) | H3 model definitions and component loading |
| Larry H3 Turbo LoRA | [Source model](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | Optional 8-step acceleration |

The dependencies and model files retain their upstream licenses and notices.
No third-party model weights or source trees are bundled here. The inference code
uses the upstream model's packed-token layout and preserves H3's interleaved QKV
ordering when loading LoRAs.
