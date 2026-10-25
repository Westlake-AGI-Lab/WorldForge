"""Place a frozen H3 model on one GPU or across two GPUs."""

from types import MethodType

import torch

from .runtime import move_context


def _chunked_mlp(self, hidden):
    chunk = self._mlp_chunk_tokens
    if not chunk or hidden.shape[0] <= chunk:
        return self._unchunked_forward(hidden)
    output = torch.empty_like(hidden)
    for start in range(0, hidden.shape[0], chunk):
        output[start:start + chunk] = self._unchunked_forward(hidden[start:start + chunk])
    return output


def _remote_forward(self, hidden, **kwargs):
    output = self._original_forward(
        hidden.to(self._compute_device), **move_context(kwargs, self._compute_device)
    )
    return output.to(self._return_device) if self._return_device else output


def place_model(model, devices, mlp_chunk_tokens=0):
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices):
        raise ValueError("Use one or two distinct CUDA devices")
    if any(torch.device(d).type != "cuda" or torch.device(d).index is None for d in devices):
        raise ValueError("Specify devices as cuda:0 or cuda:0,cuda:1")
    # MLPs act independently per token; chunking bounds activation memory.
    for block in list(model.blocks) + list(model.token_refiner.blocks):
        block.mlp._mlp_chunk_tokens = mlp_chunk_tokens
        if not hasattr(block.mlp, "_unchunked_forward"):
            block.mlp._unchunked_forward = block.mlp.forward
            block.mlp.forward = MethodType(_chunked_mlp, block.mlp)
    if len(devices) == 1:
        model.to(devices[0])
        return
    if len(model.blocks) != 50:
        raise ValueError("Expected the 50-block H3 model")
    for name, child in model.named_children():
        if name != "blocks":
            child.to(devices[0])
    for index, block in enumerate(model.blocks):
        block.to(devices[0] if index < 25 else devices[1])
        if index >= 25:
            if not hasattr(block, "_original_forward"):
                block._original_forward = block.forward
                block.forward = MethodType(_remote_forward, block)
            block._compute_device = devices[1]
            block._return_device = devices[0] if index == 49 else None


def synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)
