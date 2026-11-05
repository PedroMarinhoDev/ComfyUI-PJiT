import types

import torch

from comfyui_pjit.pjit import guidance
from comfyui_pjit.pjit.model_base import PJiTModel, MEMORY_BASE


def call(apg, x, cond, uncond, t, scale=15.0, sigmas=None):
    return apg({"cond_scale": scale, "input": x, "sigma": t.reshape(1), "cond": x - cond, "cond_denoised": cond, "uncond_denoised": uncond,
                "model_options": {"transformer_options": {"sample_sigmas": sigmas}}})


def run_steps(apg, sigmas):
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    cond, uncond = torch.randn_like(x) * 0.3, torch.randn_like(x) * 0.3
    out = []
    for t in sigmas:
        out.append(call(apg, x, cond, uncond, t, sigmas=sigmas))
    return out


def test_matches_reference(reference):
    from pjit_21k import sampler as ref
    torch.manual_seed(0)
    x = torch.randn(1, 3, 1, 32, 32) * 2
    cond, uncond = torch.randn_like(x) * 0.3, torch.randn_like(x) * 0.3
    momentum = torch.zeros_like(x)
    apg = guidance.make_apg_function()
    sigmas = torch.tensor([0.5, 0.48, 0.46])
    for t in sigmas:
        x_in = x.to(torch.bfloat16).float()
        buffer = ref.MomentumBuffer(momentum=-0.75)
        buffer.running_average = momentum.to(torch.bfloat16).float()
        v = ref.adaptive_projected_guidance(pred_cond=(x_in - cond) / t, pred_other=(x_in - uncond) / t, guidance_scale=15.0, momentum_buffer=buffer, eta=0.0, rescale=10.0)
        momentum = buffer.running_average
        want = x_in - t * v
        got = x[:, :, 0] - call(apg, x[:, :, 0], cond[:, :, 0], uncond[:, :, 0], t, sigmas=sigmas)
        assert (got - want[:, :, 0]).abs().max() < 1e-5
        x = want


def test_momentum_resets_on_a_new_run():
    sigmas = torch.tensor([0.5, 0.4, 0.3])
    apg = guidance.make_apg_function()
    first = run_steps(apg, sigmas)
    again = run_steps(apg, sigmas.clone())
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    assert not torch.equal(first[0], first[1])  # the momentum did act within the run


def test_cfg_one_is_the_conditional_prediction():
    apg = guidance.make_apg_function()
    x = torch.ones(1, 3, 8, 8)
    cond = torch.zeros_like(x)
    out = apg({"cond_scale": 1.0, "cond": x - cond, "input": x, "sigma": torch.tensor([0.5]), "cond_denoised": cond, "uncond_denoised": cond * 0, "model_options": {}})
    assert torch.equal(out, x - cond)


def test_memory_estimate_grows_with_tokens():
    bf16 = types.SimpleNamespace(get_dtype_inference=lambda: torch.bfloat16)
    fp32 = types.SimpleNamespace(get_dtype_inference=lambda: torch.float32)
    small = PJiTModel.memory_required(bf16, (1, 3, 1024, 1024))
    assert small > MEMORY_BASE
    assert PJiTModel.memory_required(bf16, (2, 3, 1024, 1024)) > small
    assert PJiTModel.memory_required(bf16, (1, 3, 2048, 2048)) > small
    assert PJiTModel.memory_required(fp32, (1, 3, 1024, 1024)) - MEMORY_BASE == 2 * (small - MEMORY_BASE)
