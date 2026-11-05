# Adapted from pjit_21k/noise.py, Copyright 2026 Linum Inc., Apache License 2.0.
"""Seeded noise as an H100 SXM draws it, so a seed gives Linum's published image on any NVIDIA GPU.

`torch.randn` on CUDA lands the same normals on different pixels depending on the GPU's SM count. This is Linum's Triton
port of that kernel (Philox4x32-10 and curand's Box-Muller as nvcc compiles them). It is only needed to reproduce their
samples: the stock noise nodes draw on the CPU and are the same on any machine, just not the same as their pictures.
"""
import math

import numpy as np
import torch

_BLOCK = 256    # threads per block in torch's RNG launch
_UNROLL = 4     # normals per curand_normal4 call
# curand_globals.h constants, folded in fp32 as nvcc folds them.
_INV_2POW32 = float(np.float32(2.3283064e-10))
_INV_2POW32_2PI = float(np.float32(np.float32(2.3283064e-10) * np.float32(6.2831855)))
_INV_2POW32_HALF = float(np.float32(_INV_2POW32) / np.float32(2))
_INV_2POW32_2PI_HALF = float(np.float32(_INV_2POW32_2PI) / np.float32(2))

_KERNEL = None


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def box_muller(x, y, inv, inv_half, inv_2pi, inv_2pi_half):
        # curand_normal.h _curand_box_muller, device branch.
        u = tl.fma(x.to(tl.float32), inv, inv_half)
        v = tl.fma(y.to(tl.float32), inv_2pi, inv_2pi_half)
        s = tl.sqrt_rn(-2.0 * tl.extra.cuda.libdevice.log(u))
        sin_v = tl.inline_asm_elementwise("sin.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        cos_v = tl.inline_asm_elementwise("cos.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        return sin_v * s, cos_v * s

    @triton.jit
    def randn_kernel(out_ptr, seed, counter0, threads, numel, iterations,
                     inv, inv_half, inv_2pi, inv_2pi_half,
                     BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = idx < threads
        zero = tl.zeros([BLOCK], dtype=tl.uint32)
        subsequence_lo = (idx & 0xFFFFFFFF).to(tl.uint32)
        subsequence_hi = (idx >> 32).to(tl.uint32)
        for it in range(iterations):
            c0 = zero + (counter0 + it).to(tl.uint32)
            r0, r1, r2, r3 = tl.philox(seed, c0, zero, subsequence_lo, subsequence_hi)
            n0, n1 = box_muller(r0, r1, inv, inv_half, inv_2pi, inv_2pi_half)
            n2, n3 = box_muller(r2, r3, inv, inv_half, inv_2pi, inv_2pi_half)
            base = idx + threads * UNROLL * it
            tl.store(out_ptr + base, n0, mask=live & (base < numel))
            tl.store(out_ptr + base + threads, n1, mask=live & (base + threads < numel))
            tl.store(out_ptr + base + 2 * threads, n2, mask=live & (base + 2 * threads < numel))
            tl.store(out_ptr + base + 3 * threads, n3, mask=live & (base + 3 * threads < numel))

    return randn_kernel


def randn_as_on_sm_count(size, generator, sm_count, max_threads_per_sm=2048):
    """`torch.randn(size, generator=generator)` in fp32 as a CUDA GPU with `sm_count` SMs would draw it."""
    global _KERNEL
    if generator.device.type != "cuda":
        raise ValueError("P-JiT: the H100 noise kernel needs a CUDA generator")
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    numel = math.prod(size)
    grid = min(math.ceil(numel / _BLOCK), max_threads_per_sm // _BLOCK * sm_count)
    threads = _BLOCK * grid
    iterations = (numel - 1) // (threads * _UNROLL) + 1
    offset = generator.get_offset()
    if offset % 4 != 0:
        raise ValueError(f"Generator offset {offset} is not a multiple of 4.")

    out = torch.empty(numel, dtype=torch.float32, device=generator.device)
    with torch.cuda.device(generator.device):
        _KERNEL[(grid,)](out, generator.initial_seed(), offset // 4, threads, numel, iterations,
                         _INV_2POW32, _INV_2POW32_HALF, _INV_2POW32_2PI, _INV_2POW32_2PI_HALF,
                         BLOCK=_BLOCK, UNROLL=_UNROLL)
    generator.set_offset(offset + iterations * 4)  # torch's counter_offset
    return out.view(tuple(size))


class H100Noise:
    """A NOISE for SamplerCustomAdvanced. Each image in a batch gets its own stream, seed + index."""

    def __init__(self, seed, sm_count=132):
        self.seed = seed
        self.sm_count = sm_count

    def generate_noise(self, input_latent):
        import comfy.model_management
        latent = input_latent["samples"]
        batch_inds = input_latent.get("batch_index", None) or list(range(latent.shape[0]))
        device = comfy.model_management.get_torch_device()
        rows = []
        for index in batch_inds:
            generator = torch.Generator(device).manual_seed((int(self.seed) + int(index)) % (1 << 32))
            rows.append(randn_as_on_sm_count(latent.shape[1:], generator, self.sm_count).cpu())
        return torch.stack(rows)
