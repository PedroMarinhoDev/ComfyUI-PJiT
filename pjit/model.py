"""The P-JiT DiT on ComfyUI's ops: a pixel-space text-to-image transformer that predicts the clean image.

Module and parameter names are Linum's (`pjit_21k/model.py`), so the released `model.safetensors` loads with no key
map. The dtype flow is theirs too, written out instead of left to autocast: the Linears and the patch embedding run
in the compute dtype (bf16), every RMSNorm normalizes in fp32 and hands fp32 tokens onward, RoPE and the output
LayerNorm stay in fp32, and attention runs in bf16. The norm gains and the output modulation are held in fp32 whatever the
file's type, so a bf16 or int8 file only changes the Linears and convs, which run in bf16 anyway.

Differences from the reference, all of which leave the values the same up to GPU rounding:
* no caption padding or masks. Cond and uncond have different lengths, so ComfyUI runs them as separate forwards
  (`model_base.PJiTModel.extra_conds`), which is what the reference's FlashAttention-3 path packs its rows down to.
* ComfyUI's own attention (`--use-sage-attention` etc. apply) and dynamic VRAM with layer prefetch.
* captions sit at the identity rotation, so the text refiner skips RoPE (a rotation by 0 is exact).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.modules.attention
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension


class RMSNorm(nn.Module):
    """Normalizes in fp32, rounds to the input dtype, then applies the fp32 gain, so the output is fp32 either way."""

    def __init__(self, dim, eps, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=torch.float32))

    def forward(self, x):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * comfy.model_management.cast_to(self.weight, dtype=torch.float32, device=x.device)


def apply_rope(x, cos, sin):
    """Rotates q or k [B, L, heads, head_dim] by the [L, 1, head_dim / 2] tables, over interleaved channel pairs."""
    b, seq_len, n, head_dim = x.shape
    pairs = x.reshape(b, seq_len, n, head_dim // 2, 2)
    x_even, x_odd = pairs[..., 0], pairs[..., 1]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(3)


class SelfAttention(nn.Module):
    """RMSNorm on q/k, 3-D RoPE, and a per-head sigmoid gate on the output."""

    def __init__(self, dim, num_heads, eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = comfy.ldm.modules.attention.ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.o = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.gate_proj = operations.Linear(dim, num_heads, device=device, dtype=dtype)

    def forward(self, x, rope, compute_dtype, transformer_options):
        b, seq_len = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        x_c = x.to(compute_dtype)
        q = self.norm_q(self.q(x_c)).view(b, seq_len, n, d)
        k = self.norm_k(self.k(x_c)).view(b, seq_len, n, d)
        v = self.v(x_c).view(b, seq_len, n, d)
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        gate = torch.sigmoid(self.gate_proj(x_c)).unsqueeze(-1)
        del x_c
        q = comfy.ldm.modules.attention.AttentionTensorContainer(q.to(compute_dtype).transpose(1, 2))
        k = comfy.ldm.modules.attention.AttentionTensorContainer(k.to(compute_dtype).transpose(1, 2))
        v = comfy.ldm.modules.attention.AttentionTensorContainer(v.transpose(1, 2))
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, n, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options,
            preferred_attention=self.comfy_attention)
        out = out.transpose(1, 2).float() * gate
        return self.o(out.flatten(2).to(compute_dtype))


class SwiGLUFFN(nn.Module):
    """w2(silu(w1 x) * w3 x), with w1/w3 packed into one Linear. The hidden width is 2/3 of `hidden_dim` rounded up
    to `multiple_of` (7936 -> 5376, 4096 -> 2816)."""

    def __init__(self, dim, hidden_dim, multiple_of=256, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w13 = operations.Linear(dim, 2 * hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x3)


class Block(nn.Module):
    """Sandwich RMSNorm with low-rank AdaLN (scale-in, tanh gate-out, no shift)."""

    def __init__(self, dim, ffn_dim, num_heads, eps, adaln_rank, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.attn = SelfAttention(dim, num_heads, eps, device=device, dtype=dtype, operations=operations)
        self.post_norm1 = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.norm2 = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(dim, ffn_dim, device=device, dtype=dtype, operations=operations)
        self.post_norm2 = RMSNorm(dim, eps, device=device, dtype=dtype)
        self.adaln_up = operations.Linear(adaln_rank, 4 * dim, device=device, dtype=dtype)

    def forward(self, x, c, rope, compute_dtype, transformer_options):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_up(c).unsqueeze(1).chunk(4, dim=2)
        y = self.attn(self.norm1(x) * (1.0 + scale_msa), rope, compute_dtype, transformer_options)
        x = x + gate_msa.tanh() * self.post_norm1(y)
        y = self.ffn((self.norm2(x) * (1.0 + scale_mlp)).to(compute_dtype))
        return x + gate_mlp.tanh() * self.post_norm2(y)


class BottleneckPatchEmbed(nn.Module):
    """A patchifying Conv3d into `bottleneck_dim` channels, then a 1x1 Conv3d up to the model width."""

    def __init__(self, in_channels, dim, patch, bottleneck_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.proj1 = operations.Conv3d(in_channels, bottleneck_dim, kernel_size=patch, stride=patch, bias=False, device=device, dtype=dtype)
        self.proj2 = operations.Conv3d(bottleneck_dim, dim, kernel_size=1, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class OutputHead(nn.Module):
    """AdaLN-modulated LayerNorm (fp32, as autocast runs it) and a linear projection to pixels."""

    def __init__(self, dim, out_channels, patch, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm = operations.LayerNorm(dim, device=device, dtype=torch.float32)
        self.proj = operations.Linear(dim, math.prod(patch) * out_channels, device=device, dtype=dtype)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=torch.float32))

    def forward(self, x, e, compute_dtype):
        modulation = comfy.model_management.cast_to(self.modulation, dtype=torch.float32, device=x.device)
        shift, scale = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.proj((self.norm(x.float()) * (1 + scale) + shift).to(compute_dtype))


def sinusoidal_embedding_1d(dim, position, dtype):
    """[cos | sin] timestep embedding, computed in fp64."""
    half = dim // 2
    position = position.to(torch.float64)
    inv_freq = torch.pow(10000.0, -torch.arange(half, device=position.device).to(position) / half)
    angle = torch.outer(position, inv_freq)
    return torch.cat((torch.cos(angle), torch.sin(angle)), dim=1).to(dtype)


def rotary_table(positions, dim, theta=10000.0):
    """cos and sin of p * theta^(-2k/dim) in fp64, for one axis, [P, dim / 2] each."""
    inv_freq = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float64), torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    angle = torch.outer(positions.to(torch.float64), inv_freq)
    return angle.cos(), angle.sin()


class PJiTDiT(nn.Module):
    def __init__(self, dim=2944, num_heads=23, num_layers=22, ffn_dim=7936, refiner_blocks=2, refiner_ffn_dim=4096,
                 adaln_rank=256, text_dim=7680, in_channels=3, out_channels=3, freq_dim=256, eps=1e-6,
                 bottleneck_dim=1024, patch=(1, 32, 32), rope_max_positions=1024, rope_spatial_scale=0.5,
                 image_model=None, compute_dtype=torch.bfloat16, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.patch = tuple(patch)
        self.out_channels = out_channels
        self.freq_dim = freq_dim
        self.head_dim = dim // num_heads
        self.rope_max_positions = rope_max_positions
        self.rope_spatial_scale = rope_spatial_scale
        self._rope_cache = {}

        ops = dict(device=device, dtype=dtype, operations=operations)
        self.patch_embed = BottleneckPatchEmbed(in_channels, dim, self.patch, bottleneck_dim, **ops)
        self.time_embed = nn.Sequential(
            operations.Linear(freq_dim, dim, device=device, dtype=dtype), nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype))
        self.image_adaln = nn.Sequential(nn.SiLU(), operations.Linear(dim, adaln_rank, device=device, dtype=dtype))
        self.text_adaln = nn.Sequential(nn.SiLU(), operations.Linear(dim, adaln_rank, device=device, dtype=dtype))
        self.text_proj = nn.Sequential(
            operations.Linear(text_dim, dim, bias=False, device=device, dtype=dtype), nn.GELU(approximate="tanh"),
            operations.Linear(dim, dim, bias=False, device=device, dtype=dtype))
        block = dict(num_heads=num_heads, eps=eps, adaln_rank=adaln_rank, **ops)
        self.image_refiner = nn.ModuleList([Block(dim, refiner_ffn_dim, **block) for _ in range(refiner_blocks)])
        self.text_refiner = nn.ModuleList([Block(dim, refiner_ffn_dim, **block) for _ in range(refiner_blocks)])
        self.blocks = nn.ModuleList([Block(dim, ffn_dim, **block) for _ in range(num_layers)])
        self.head = OutputHead(dim, out_channels, self.patch, **ops)

    def rope_tables(self, h_p, w_p, text_len, device):
        """Image tokens rotate by (t = 0, h * 0.5, w * 0.5) in a (22, 21, 21) split of the 64 pairs; caption tokens by 0."""
        key = (h_p, w_p, device)
        table = self._rope_cache.get(key)
        if table is None:
            if max(h_p, w_p) > self.rope_max_positions:
                raise ValueError(f"P-JiT: a {h_p}x{w_p} patch grid exceeds the {self.rope_max_positions}-position RoPE table")
            side = self.head_dim // 6
            temporal = self.head_dim - 4 * side
            cos_t, sin_t = rotary_table(torch.zeros(1), temporal)
            cos_h, sin_h = rotary_table(torch.arange(h_p) * self.rope_spatial_scale, 2 * side)
            cos_w, sin_w = rotary_table(torch.arange(w_p) * self.rope_spatial_scale, 2 * side)

            def grid(t, h, w):
                return torch.cat((t.view(1, 1, -1).expand(h_p, w_p, -1),
                                  h.view(h_p, 1, -1).expand(h_p, w_p, -1),
                                  w.view(1, w_p, -1).expand(h_p, w_p, -1)), dim=-1).reshape(h_p * w_p, 1, self.head_dim // 2)

            table = (grid(cos_t, cos_h, cos_w).to(device, torch.float32), grid(sin_t, sin_h, sin_w).to(device, torch.float32))
            self._rope_cache = {key: table}  # one grid at a time: a batch of sizes is not a thing here
        cos, sin = table
        if text_len == 0:
            return cos, sin
        return torch.cat((cos, cos.new_ones((text_len, 1, cos.shape[-1]))), dim=0), torch.cat((sin, sin.new_zeros((text_len, 1, sin.shape[-1]))), dim=0)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        """x [B, 3, H, W] in [-1, 1] noise-space, timestep = t * 1000, context [B, L, 7680]. Returns the clean image."""
        compute_dtype = self.compute_dtype
        device = x.device
        h_in, w_in = x.shape[-2:]
        _, patch_h, patch_w = self.patch
        pad_h, pad_w = (patch_h - h_in % patch_h) % patch_h, (patch_w - w_in % patch_w) % patch_w
        if pad_h or pad_w:  # as in training, e.g. 1280x720 -> 1280x736, cropped back below
            x = F.pad(x, (0, pad_w, 0, pad_h))

        tokens = self.patch_embed(x.unsqueeze(2).to(compute_dtype))
        h_p, w_p = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2).contiguous()
        l_img = tokens.shape[1]

        # the reference truncates t * 1000 to an integer, off by one at some steps of fp32 linspace: part of the contract
        e = self.time_embed(sinusoidal_embedding_1d(self.freq_dim, timestep.long(), compute_dtype))
        c_image = self.image_adaln(e)
        c_text = self.text_adaln(e)
        text = self.text_proj(context.to(compute_dtype))

        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
            list(self.image_refiner) + list(self.text_refiner) + list(self.blocks), device, transformer_options)

        rope = self.rope_tables(h_p, w_p, 0, device)
        for block in self.image_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, compute_dtype)
            tokens = block(tokens, c_image, rope, compute_dtype, transformer_options)
        for block in self.text_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, compute_dtype)
            text = block(text, c_text, None, compute_dtype, transformer_options)

        seq = torch.cat((tokens, text), dim=1)
        del tokens, text
        rope = self.rope_tables(h_p, w_p, seq.shape[1] - l_img, device)
        for block in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, compute_dtype)
            seq = block(seq, c_image, rope, compute_dtype, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        out = self.head(seq[:, :l_img], e, compute_dtype)
        del seq
        b = out.shape[0]
        out = out.view(b, h_p, w_p, patch_h, patch_w, self.out_channels)
        out = torch.einsum("bhwqrc->bchqwr", out).reshape(b, self.out_channels, h_p * patch_h, w_p * patch_w)
        return out[:, :, :h_in, :w_in]
