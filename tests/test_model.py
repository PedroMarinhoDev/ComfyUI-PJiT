import importlib.util
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

import comfy.ops
from comfyui_pjit.pjit import loader
from comfyui_pjit.pjit.model import PJiTDiT, sinusoidal_embedding_1d

TINY = dict(dim=8 * 23, num_heads=23, num_layers=2, ffn_dim=128, refiner_blocks=1, refiner_ffn_dim=96, adaln_rank=16, text_dim=24, freq_dim=32, bottleneck_dim=32)


def tiny_native(dtype=torch.float32):
    torch.manual_seed(0)
    model = PJiTDiT(**TINY, dtype=dtype, device="cpu", operations=comfy.ops.manual_cast)
    for name, parameter in model.named_parameters():
        if name.endswith("modulation"):
            parameter.data = torch.randn_like(parameter) * 0.01
        elif name.endswith("weight") and parameter.ndim == 1:
            parameter.data = torch.ones_like(parameter)
        elif parameter.ndim == 1:
            parameter.data = torch.zeros_like(parameter)
        else:
            parameter.data = torch.randn_like(parameter) * 0.02
    return model


def test_forward_keeps_size_and_pads_to_the_patch():
    model = tiny_native()
    for height, width in ((64, 64), (40, 72), (33, 95)):
        out = model(torch.randn(2, 3, height, width), torch.tensor([500.0, 980.0]), context=torch.randn(2, 5, 24), transformer_options={})
        assert out.shape == (2, 3, height, width)
        assert out.dtype == torch.bfloat16


def test_rope_rows_split_as_the_reference():
    model = tiny_native()
    cos, sin = model.rope_tables(2, 3, 4, torch.device("cpu"))
    assert cos.shape == (2 * 3 + 4, 1, 4)
    assert torch.equal(cos[6:], torch.ones(4, 1, 4)) and torch.equal(sin[6:], torch.zeros(4, 1, 4))
    assert torch.equal(sin[0], torch.zeros(1, 4))  # token (0, 0) sits at the identity


def test_integer_timesteps_are_truncated():
    a = sinusoidal_embedding_1d(16, torch.tensor([459.9]).long(), torch.float32)
    b = sinusoidal_embedding_1d(16, torch.tensor([459]), torch.float32)
    assert torch.equal(a, b)


def test_matches_reference_on_cpu(reference):
    """Same weights and inputs, the reference padded and masked under bf16 autocast, the port unpadded. Bf16-level agreement."""
    from pjit_21k.config import PyramidJiTConfig
    from pjit_21k.model import PyramidJiT
    config = PyramidJiTConfig(text_len=8, rope_max_positions=64, **TINY)
    ref = PyramidJiT(config).eval()
    native = tiny_native()
    missing = ref.load_state_dict(native.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys
    torch.manual_seed(1)
    x = torch.randn(1, 3, 64, 96)
    text = torch.randn(1, 5, 24)
    t = torch.tensor([500])
    padded = torch.cat((text, torch.zeros(1, 3, 24)), dim=1)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        want = ref(x_t=x.unsqueeze(2), text=padded, text_lens=torch.tensor([5], dtype=torch.int32), t=t)[:, :, 0].float()
    with torch.inference_mode():
        got = native(x.to(torch.bfloat16), t.float(), context=text.to(torch.bfloat16), transformer_options={}).float()
    assert ((got - want).norm() / want.norm()) < 0.02


def test_config_reads_the_released_shapes(dit_file):
    with open(dit_file, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    state = {k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}
    config = loader.config_from_state_dict(state)
    assert config["dim"] == 2944 and config["num_heads"] == 23 and config["num_layers"] == 22 and config["patch"] == (1, 32, 32)
    assert config["text_dim"] == 7680 and config["adaln_rank"] == 256 and config["bottleneck_dim"] == 1024
    with torch.device("meta"):
        model = PJiTDiT(**config, dtype=torch.float32, operations=comfy.ops.manual_cast)
    keys = set(model.state_dict())
    assert keys == set(state), f"{sorted(keys ^ set(state))[:8]}"
    assert all(model.state_dict()[k].shape == state[k].shape for k in keys)


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_keeps_the_loaders_fp32_tensors():
    tool = _load_tool("convert_pjit")
    with torch.device("meta"):
        keys = list(PJiTDiT(**TINY, dtype=torch.float32, operations=comfy.ops.manual_cast).state_dict())
    assert [k for k in keys if tool.keeps_fp32(k)] == [k for k in keys if loader.keeps_fp32(k)]
    kept = {k.split(".")[-2] for k in keys if loader.keeps_fp32(k)}
    assert kept == {"norm1", "norm2", "post_norm1", "post_norm2", "norm_q", "norm_k", "norm", "head"}


def test_bf16_and_int8_files_load_and_run(tmp_path):
    """A tiny model through tools/convert_pjit.py: bf16 gives the fp32 file's output exactly (the norms stay fp32 and
    the Linears run in bf16 either way); int8 ConvRot loads into ComfyUI's quantized Linears and stays close."""
    tiny = dict(TINY, dim=16 * 23)  # a width the ConvRot groups divide
    torch.manual_seed(0)
    model = PJiTDiT(**tiny, dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    for name, parameter in model.named_parameters():
        parameter.data = torch.ones_like(parameter) if name.endswith("weight") and parameter.ndim == 1 else torch.randn_like(parameter) * 0.02
    source = tmp_path / "fp32.safetensors"
    save_file(model.state_dict(), str(source))
    paths = _load_tool("convert_pjit").convert(str(source), str(tmp_path), name="tiny", log=lambda *a: None)

    x, t, text = torch.randn(1, 3, 64, 96), torch.tensor([700.0]), torch.randn(1, 5, 24)

    def run(path, weight_dtype="default"):
        # bf16 compute as on a GPU that has it; on the CPU the loader picks fp32 by itself
        diffusion_model = loader.load_dit(path, weight_dtype=weight_dtype, apg=False, compute_dtype=torch.bfloat16).model.diffusion_model
        quantized = type(diffusion_model.blocks[0].attn.q.weight).__name__ == "QuantizedTensor"
        assert quantized == path.endswith("int8_convrot.safetensors") and diffusion_model.blocks[0].norm1.weight.dtype == torch.float32
        with torch.inference_mode():
            return diffusion_model(x.to(torch.bfloat16), t, context=text.to(torch.bfloat16), transformer_options={}).float()

    full = run(str(source))
    assert torch.equal(run(paths["bf16"]), full)
    assert torch.equal(run(str(source), "bf16"), full)
    int8 = run(paths["int8_convrot"])
    assert ((int8 - full).norm() / full.norm()) < 0.05
    assert loader.load_dit(paths["bf16"], apg=False).model.diffusion_model.compute_dtype == torch.float32
