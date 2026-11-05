#!/usr/bin/env python3
"""Write the bf16 and int8 ConvRot files from Linum's fp32 `model.safetensors`.

    python tools/convert_pjit.py ~/AI/hf_originals/pjit-21k/model.safetensors ComfyUI/models/diffusion_models

bf16 (`pjit_21k_bf16.safetensors`, 4.4 GB): the Linear and conv weights and biases in bf16, which is what the
reference's bf16 autocast feeds its matmuls, and the norm gains and the output modulation in fp32, as it applies them.
It gives the same pictures as the fp32 file.

int8 ConvRot (`pjit_21k_int8_convrot.safetensors`, 2.3 GB): as the bf16 file, plus q, k, v, o, w13 and w2 of every
block and refiner in ComfyUI's `int8_tensorwise` format with ConvRot: the weight rotated by a Hadamard matrix in groups
along its input, then one int8 scale per output row, quantized from the fp32 weights. ComfyUI's Hadamard sizes are
powers of 4, so the group is 256 where the input width allows it and 64 for the 2944-wide inputs. Run with ComfyUI's Python, which has comfy-kitchen.
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANTIZED = ("attn.q", "attn.k", "attn.v", "attn.o", "ffn.w13", "ffn.w2")
STACKS = ("blocks.", "image_refiner.", "text_refiner.")
METADATA = {"format": "pt", "model": "P-JiT 21K", "weights": "post-trained", "source": "Linum-AI/pjit-21k model.safetensors"}


def keeps_fp32(key):
    """The same rule as pjit/loader.py: norm gains and the output modulation stay fp32."""
    parts = key.split(".")
    return key == "head.modulation" or (len(parts) > 1 and parts[-2].startswith(("norm", "post_norm")))


def quantized_layer(key):
    """The Linear a weight belongs to, if it is one the int8 file quantizes."""
    if not key.startswith(STACKS) or not key.endswith(".weight"):
        return None
    layer = key[: -len(".weight")]
    return layer if layer.split(".", 2)[2] in QUANTIZED else None


def group_size(in_features):
    for size in (256, 64, 16):
        if in_features % size == 0:
            return size
    raise ValueError(f"no ConvRot group divides an input width of {in_features}")


def convert(source_path, out_dir, name="pjit_21k", device="cpu", log=print):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    start = time.time()
    source = safe_open(source_path, "pt", device="cpu")
    bf16, int8 = {}, {}
    worst = (0.0, "")
    for key in source.keys():
        tensor = source.get_tensor(key)
        if tensor.dtype != torch.float32:
            raise SystemExit(f"{key} is {tensor.dtype}; this expects Linum's fp32 release")
        kept = tensor if keeps_fp32(key) else tensor.to(torch.bfloat16)
        bf16[key] = kept
        layer = quantized_layer(key)
        if layer is None:
            int8[key] = kept
            continue
        group = group_size(tensor.shape[1])
        weight = tensor.to(device)
        qdata, params = TensorWiseINT8Layout.quantize(weight, is_weight=True, per_channel=True, convrot=True,
                                                      convrot_groupsize=group)
        restored = TensorWiseINT8Layout.dequantize(qdata, params).float()
        worst = max(worst, (((restored - weight).norm() / weight.norm()).item(), key))
        int8[layer + ".weight"] = qdata.cpu()
        int8[layer + ".weight_scale"] = params.scale.reshape(-1, 1).float().cpu()
        meta = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
        int8[layer + ".comfy_quant"] = torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for suffix, tensors in (("bf16", bf16), ("int8_convrot", int8)):
        path = os.path.join(out_dir, f"{name}_{suffix}.safetensors")
        save_file(tensors, path + ".part", metadata=dict(METADATA, dtype=suffix))
        os.replace(path + ".part", path)
        paths[suffix] = path
        log(f"{path}: {len(tensors)} tensors, {os.path.getsize(path) / 1e9:.2f} GB")
    log(f"largest relative weight error after int8: {worst[0]:.4f} ({worst[1]}); {time.time() - start:.0f} s")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Linum's fp32 model.safetensors")
    parser.add_argument("out_dir")
    parser.add_argument("--name", default="pjit_21k")
    parser.add_argument("--device", default="cpu", help="where to rotate the weights: cpu, or cuda:N")
    args = parser.parse_args()
    convert(args.source, args.out_dir, args.name, args.device)


if __name__ == "__main__":
    main()
