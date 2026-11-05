"""Load the P-JiT DiT into a native ComfyUI MODEL.

The file is detected from its tensors, not its name. The model is built the way the other native packs here build
theirs: a `BaseModel` subclass, the patcher class looked up at call time (main.py swaps `CoreModelPatcher` for the dynamic
one), `assign=` loading so dynamic VRAM streams the weights from the file's pages, and `cached_patcher_init` so Select
Model Device can rebuild it.
"""
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .guidance import make_apg_function
from .model_base import PJiTModel, PJiTModelConfig

SIGNATURE = ("patch_embed.proj1.weight", "image_adaln.1.weight", "text_proj.0.weight", "head.modulation")
WEIGHT_DTYPES = {"default": None, "bf16": torch.bfloat16, "fp32": torch.float32}


def config_from_state_dict(state_dict):
    """The architecture, read from the weights, so a wrong file fails here and not in the middle of a forward."""
    missing = [k for k in SIGNATURE if k not in state_dict]
    if missing:
        raise ValueError(f"not a P-JiT checkpoint (no {missing[0]})")
    proj1 = state_dict["patch_embed.proj1.weight"].shape
    dim = state_dict["patch_embed.proj2.weight"].shape[0]
    return {
        "dim": dim,
        "num_heads": state_dict["blocks.0.attn.gate_proj.weight"].shape[0],
        "num_layers": 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        "refiner_blocks": 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("image_refiner.")),
        # the FFN's hidden width comes back through the 2/3 + round-up in SwiGLUFFN
        "ffn_dim": state_dict["blocks.0.ffn.w13.weight"].shape[0] * 3 // 4,
        "refiner_ffn_dim": state_dict["image_refiner.0.ffn.w13.weight"].shape[0] * 3 // 4,
        "adaln_rank": state_dict["image_adaln.1.weight"].shape[0],
        "text_dim": state_dict["text_proj.0.weight"].shape[1],
        "in_channels": proj1[1],
        "out_channels": state_dict["head.proj.weight"].shape[0] // (proj1[2] * proj1[3] * proj1[4]),
        "freq_dim": state_dict["time_embed.0.weight"].shape[1],
        "bottleneck_dim": proj1[0],
        "patch": tuple(proj1[2:5]),
    }


def keeps_fp32(key):
    """The RMSNorm and LayerNorm gains and the output modulation, which the reference applies in fp32: they stay fp32
    in every file and at every weight_dtype. Everything else feeds a bf16 Linear or conv."""
    parts = key.split(".")
    return key == "head.modulation" or (len(parts) > 1 and parts[-2].startswith(("norm", "post_norm")))


def _factory(path, weight_dtype="default", apg=True, disable_dynamic=False):
    return load_dit(path, weight_dtype=weight_dtype, apg=apg, disable_dynamic=disable_dynamic)


def load_dit(path, weight_dtype="default", apg=True, disable_dynamic=False, compute_dtype=None):
    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    unet_config = config_from_state_dict(state_dict)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    if compute_dtype is None:
        # bf16 as trained, where the device has it (RTX 30 and newer, RDNA3, Apple on macOS 14); fp32 elsewhere, since
        # bf16 is emulated there and fp16 is not a dtype the model was checked in. --force-fp32 also lands here.
        compute_dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(load_device) else torch.float32

    quant_config = comfy.utils.detect_layer_quantization(state_dict, "")
    if quant_config:
        # int8 ConvRot (tools/convert_pjit.py): ComfyUI's own quantized Linears, dequantized where the device has no int8
        if weight_dtype != "default":
            logging.warning("P-JiT: weight_dtype %s is ignored for the quantized file %s", weight_dtype, os.path.basename(path))
        dtype = compute_dtype
        operations = comfy.ops.mixed_precision_ops(quant_config, compute_dtype,
                                                   disabled=comfy.ops.get_disabled_quant_formats(load_device))
        kind = "int8 ConvRot"
    else:
        dtype = WEIGHT_DTYPES[weight_dtype] or state_dict["blocks.0.attn.q.weight"].dtype
        state_dict = {k: v if keeps_fp32(k) or v.dtype == dtype else v.to(dtype) for k, v in state_dict.items()}
        operations = comfy.ops.manual_cast
        kind = f"{dtype} weights"

    model_config = PJiTModelConfig(unet_config, dtype, compute_dtype, operations=operations, quant_config=quant_config)
    model = PJiTModel(model_config, device=offload_device)

    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
    patcher.cached_patcher_init = (_factory, (path, weight_dtype, apg))
    if apg:
        patcher.set_model_sampler_cfg_function(make_apg_function())

    loaded = model.diffusion_model.load_state_dict(state_dict, strict=False, assign=patcher.is_dynamic())
    missing, unexpected = list(loaded.missing_keys), list(loaded.unexpected_keys)
    del state_dict
    if missing or unexpected:
        raise ValueError(f"P-JiT checkpoint mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
                         f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})")
    logging.info("P-JiT: loaded %s (%s, %s compute, %d blocks, dim %d)", os.path.basename(path), kind, compute_dtype,
                 unet_config["num_layers"], unet_config["dim"])
    return patcher
