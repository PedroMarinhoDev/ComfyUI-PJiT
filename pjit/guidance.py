"""Adaptive projected guidance as the reference runs it (`pjit_21k/sampler.py`, Sadat et al., arXiv:2410.02416).

The stock `APG` node is a different recipe for this model: it works on x_0 differences, while P-JiT's runs on velocities,
so its norm clip of 10 and the projection see other numbers (x_0 differences are the velocity's times t).

The rounding follows the reference. The model saw the state rounded to bf16, so the velocities are taken from that
state, and the momentum buffer is rounded to bf16 before each step uses it.
"""
import math

import torch


def project(v0, v1):
    """v0's parts along and across v1, per sample, in fp64."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    dims = list(range(1, v1.ndim))
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    return parallel.to(dtype), (v0 - parallel).to(dtype)


def apg_velocity(v_cond, v_uncond, scale, momentum_state, momentum, eta, rescale):
    """One guided velocity. Returns it with the new momentum state (before the norm clip, as the reference keeps it)."""
    diff = v_cond - v_uncond
    if momentum_state is not None:
        diff = diff + momentum * momentum_state.to(torch.bfloat16).float()
    new_state = diff
    if rescale > 0:
        norm = diff.flatten(1).norm(p=2, dim=1).view(-1, *([1] * (diff.ndim - 1)))
        diff = diff * torch.minimum(torch.ones_like(diff), rescale / norm)
    parallel, orthogonal = project(diff, v_cond)
    return v_cond + scale * (orthogonal + eta * parallel), new_state


def make_apg_function(momentum=-0.75, eta=0.0, rescale=10.0):
    """A `sampler_cfg_function` with the momentum kept across the steps of one sampling run."""
    state = {"momentum": None, "run": None, "sigma": None, "shape": None}

    def apg(args):
        cond_scale = args["cond_scale"]
        if math.isclose(cond_scale, 1.0):  # the sampler skipped the uncond pass
            return args["cond"]
        x, sigma = args["input"], args["sigma"]
        run = args["model_options"].get("transformer_options", {}).get("sample_sigmas", None)
        sigma_now = float(sigma.max())
        if state["run"] is not run or state["shape"] != x.shape or sigma_now > state["sigma"]:
            state["momentum"] = None
        state["run"], state["sigma"], state["shape"] = run, sigma_now, x.shape

        t = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        x_in = x.to(torch.bfloat16).float()
        v_cond = (x_in - args["cond_denoised"]) / t
        v_uncond = (x_in - args["uncond_denoised"]) / t
        v_guided, state["momentum"] = apg_velocity(v_cond, v_uncond, cond_scale, state["momentum"], momentum, eta, rescale)
        return x - (x_in - t * v_guided)

    return apg
