"""P-JiT as a native ComfyUI `BaseModel`, driven by the stock samplers.

The sampling contract (`pjit_21k/sampler.py`): flow matching with x_t = (1 - t) x_0 + t * 4 * n, 50 uniform Euler steps
from t = 1 to t = 0, APG at scale 15 against an empty negative prompt (`guidance.py`). In ComfyUI that is
`ModelType.FLOW` (`ModelSamplingDiscreteFlow` + `CONST`, whose `noise_scale` setting multiplies the noise), sampler
`euler`, scheduler `simple` or the pack's uniform one, cfg 15.

There is no latent space. A LATENT here is the image itself, [B, 3, H, W] in [-1, 1]: the stock `pixel_space` VAE
converts to and from IMAGE, and Chroma Radiance's latent format already describes it (3 channels, downscale 1).
"""

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.ops

from .model import PJiTDiT

NOISE_SCALE = 4.0
MAX_TEXT_TOKENS = 256
MEMORY_PER_TOKEN = 128 * 1024
MEMORY_BASE = 256 * 1024 * 1024


class PJiTModelConfig:
    def __init__(self, unet_config, weight_dtype, compute_dtype, operations=None, quant_config=None):
        self.unet_config = dict(unet_config, dtype=weight_dtype, compute_dtype=compute_dtype)
        self.latent_format = comfy.latent_formats.ChromaRadiance()
        self.sampling_settings = {"shift": 1.0, "multiplier": 1000, "noise_scale": NOISE_SCALE}
        self.quant_config = quant_config
        # manual_cast even for bf16 weights: the output LayerNorm and the fp32 norms read their weights in fp32.
        # Quantized files get mixed_precision_ops, which is manual_cast plus the quantized Linear.
        self.manual_cast_dtype = compute_dtype
        self.custom_operations = operations or comfy.ops.manual_cast
        self.optimizations = {}
        self.model_type = None
        self.memory_usage_factor = 1.0
        self.supported_inference_dtypes = [compute_dtype]


class PJiTModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, comfy.model_base.ModelType.FLOW, device=device, unet_model=PJiTDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            # no repeat padding: the DiT attends to every caption token, so captions of different lengths run separately
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        return out

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        # The DiT predicts x_0 directly, so this returns its output instead of going through the velocity formula of
        # `calculate_denoised`: x_0 stays exact, and a stock ModelSampling node on top only moves the schedule.
        dtype = self.get_dtype_inference()
        xc = x.to(dtype)
        timestep = self.model_sampling.timestep(t).float()
        context = comfy.model_management.cast_to_device(c_crossattn, xc.device, dtype)
        transformer_options = transformer_options.copy()
        transformer_options["prefetch_dynamic_vbars"] = self.current_patcher is not None and self.current_patcher.is_dynamic()
        return self.diffusion_model(xc, timestep, context=context, control=control, transformer_options=transformer_options).float()

    def memory_required(self, input_shape, cond_shapes={}):
        # measured on an RTX 3090: 87 KiB per token, over a floor of about 40 MiB of cast weights. Held with some margin.
        batch, height, width = input_shape[0], input_shape[-2], input_shape[-1]
        tokens = batch * (((height + 31) // 32) * ((width + 31) // 32) + MAX_TEXT_TOKENS)
        scale = self.get_dtype_inference().itemsize / 2  # measured in bf16; fp32 compute holds twice as much
        return int(tokens * MEMORY_PER_TOKEN * scale) + MEMORY_BASE
