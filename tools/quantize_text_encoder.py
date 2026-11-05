#!/usr/bin/env python3
"""Smaller Qwen3.5-4B text encoders: int8 ConvRot, W6A8 and W4A8, on Comfy-Org's recipes for their Qwen3.5-9B files.

    python tools/quantize_text_encoder.py ComfyUI/models/text_encoders/qwen3.5_4b_bf16.safetensors ComfyUI/models/text_encoders

int8 ConvRot (`qwen3.5_4b_int8_convrot.safetensors`): every Linear of the language model, the vision tower and the
MTP head, and the embedding table (which is also the output layer, tied), in ComfyUI's `int8_tensorwise` format with
ConvRot groups of 256. Norms, the convolutions, the small gate projections and the vision merger stay bf16.

W6A8 (`qwen3.5_4b_w6a8.safetensors`) and W4A8 (`qwen3.5_4b_w4a8.safetensors`): the language model's Linears in
ComfyUI's `w6a8_int8` (uniform 6-bit codes) or `asym_w4a8_int8` (4-bit codes from a 16-level Lloyd-Max codebook), both
with fp8 scales per 16 weights and ConvRot groups of 256, rounded to nearest without calibration; the embedding table,
the vision tower and the MTP head int8 ConvRot as in the int8 file (Comfy-Org's W4A8 keeps those two in bf16; neither
is on the path to P-JiT's caption embeddings).

Run with ComfyUI's Python (comfy-kitchen). Each layer is built as ComfyUI's own quantized Linear and saved from its
state dict, so the keys and `comfy_quant` metadata are what the loader reads.
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM = re.compile(r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$")
VISUAL = re.compile(r"^model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])\.weight$")
MTP = re.compile(r"^mtp\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
EMBEDDING = "model.language_model.embed_tokens.weight"
CONVROT = 256
W4A8_GROUP = 16
GROUPED_FORMATS = {4: "asym_w4a8_int8", 6: "w6a8_int8"}
METADATA = {"format": "pt", "source": "Qwen/Qwen3.5-4B (bf16), quantized by ComfyUI-PJiT tools/quantize_text_encoder.py"}


def ops():
    from comfy.cli_args import args
    args.cpu = True  # scratch modules only: nothing runs on them
    import comfy.ops
    return comfy.ops.mixed_precision_ops({})


def int8_entries(weight):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    qdata, params = TensorWiseINT8Layout.quantize(weight.float(), is_weight=True, per_channel=True, convrot=True,
                                                  convrot_groupsize=CONVROT)
    restored = TensorWiseINT8Layout.dequantize(qdata, params).float()
    meta = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT}
    return {"weight": qdata.cpu(), "weight_scale": params.scale.reshape(-1, 1).float().cpu(),
            "comfy_quant": torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)}, restored


def grouped_entries(weight, scratch, bits):
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
    qdata, params = AsymW4A8Int8Layout.quantize(weight, group_size=W4A8_GROUP, convrot_groupsize=CONVROT, bits=bits)
    restored = AsymW4A8Int8Layout.dequantize(qdata, params).float()
    module = scratch.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16)
    module.quant_format = GROUPED_FORMATS[bits]
    module.layout_type = "AsymW4A8Int8Layout"
    module.weight = torch.nn.Parameter(QuantizedTensor(qdata.cpu(), module.layout_type, params.to_device(torch.device("cpu"))),
                                       requires_grad=False)
    return {key: value for key, value in module.state_dict().items()}, restored


def plan(key):
    """What each file does with a tensor: "int8", 6 or 4 bits, or None to keep it as it is."""
    other = "int8" if (VISUAL.match(key) or MTP.match(key) or key == EMBEDDING) else None
    language = bool(LM.match(key))
    return {"int8_convrot": "int8" if language else other, "w6a8": 6 if language else other, "w4a8": 4 if language else other}


def quantize(source_path, out_dir, device="cuda:0", log=print):
    scratch = ops()
    source = safe_open(source_path, "pt", device="cpu")
    outputs = {"int8_convrot": {}, "w6a8": {}, "w4a8": {}}
    worst = {name: (0.0, "") for name in outputs}
    start = time.time()
    for key in source.keys():
        tensor = source.get_tensor(key)
        layer = key[: -len(".weight")]
        plans = plan(key)
        if not any(plans.values()):
            for out in outputs.values():
                out[key] = tensor
            continue
        weight = tensor.to(device)
        for name, how in plans.items():
            if how is None:
                outputs[name][key] = tensor
                continue
            entries, restored = int8_entries(weight) if how == "int8" else grouped_entries(weight, scratch, how)
            error = ((restored - weight.float()).norm() / weight.float().norm()).item()
            worst[name] = max(worst[name], (error, key))
            for suffix, value in entries.items():
                outputs[name][f"{layer}.{suffix}"] = value
        del weight
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, tensors in outputs.items():
        path = os.path.join(out_dir, f"qwen3.5_4b_{name}.safetensors")
        save_file(tensors, path + ".part", metadata=METADATA)
        os.replace(path + ".part", path)
        paths[name] = path
        quantized = sum(1 for k in tensors if k.endswith(".comfy_quant"))
        log(f"{path}: {len(tensors)} tensors, {quantized} quantized layers, {os.path.getsize(path) / 1e9:.2f} GB, "
            f"largest relative weight error {worst[name][0]:.4f} ({worst[name][1]})")
    log(f"{time.time() - start:.0f} s")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="qwen3.5_4b_bf16.safetensors (Comfy-Org/Qwen3.5, or tools/merge_qwen35.py)")
    parser.add_argument("out_dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    quantize(args.source, args.out_dir, args.device)


if __name__ == "__main__":
    sys.exit(main())
