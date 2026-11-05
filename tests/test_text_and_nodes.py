import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from comfyui_pjit import nodes
from comfyui_pjit.pjit import text, upsampler

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "fixtures/tokens.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[f"{len(c['ids'])}_tokens" for c in CASES])
def test_tokens_match_the_reference(case):
    tokenizer = text.PJiTTokenizer()
    assert [t[0] for t in tokenizer.tokenize_with_weights(case["caption"])[text.MODEL_TYPE][0]] == case["ids"]


def test_generation_is_not_truncated():
    tokenizer = text.PJiTTokenizer()
    long_text = " ".join(["word"] * 1000)
    assert len(tokenizer.tokenize_with_weights(long_text, truncate=False)[text.MODEL_TYPE][0]) > 900
    assert len(tokenizer.tokenize_with_weights(long_text)[text.MODEL_TYPE][0]) == 256


def test_scheduler_is_the_reference_linspace():
    assert torch.equal(nodes.PJiTScheduler.execute(steps=50, denoise=1.0).result[0], torch.linspace(1.0, 0.0, 51))
    partial = nodes.PJiTScheduler.execute(steps=30, denoise=0.6).result[0]
    assert len(partial) == 31 and partial[-1] == 0.0 and 0.59 < partial[0] < 0.61


def test_empty_image_sizes_are_the_trained_ones():
    assert set(nodes.SIZES.values()) == {(1024, 1024), (864, 1216), (1152, 864), (1248, 832), (1280, 720)}
    assert nodes.PJiTEmptyImage.execute(size="tall 864x1216", width=1, height=1, batch_size=2).result[0]["samples"].shape == (2, 3, 1216, 864)
    assert nodes.PJiTEmptyImage.execute(size=nodes.CUSTOM_SIZE, width=640, height=360, batch_size=1).result[0]["samples"].shape == (1, 3, 360, 640)


class FakeClip:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def tokenize(self, prompt, truncate=True):
        assert truncate is False
        self.prompts.append(prompt)
        return {}

    def generate(self, tokens, **kwargs):
        return [0]

    def decode(self, ids):
        return self.replies.pop(0)


CAPTION = " ".join(["A round sourdough loaf rests on a scarred wooden board while soft window light falls across its floured crust and the kitchen behind it melts into a warm blur"] * 2)


def test_local_upsampler_accepts_untagged_and_tagged_captions():
    assert upsampler.upsample_prompt_local(FakeClip([CAPTION]), "bread") == (CAPTION, None)
    assert upsampler.upsample_prompt_local(FakeClip([f"<caption>{CAPTION}</caption>"]), "bread") == (CAPTION, None)


def test_local_upsampler_keeps_errors_and_declines():
    assert upsampler.upsample_prompt_local(FakeClip(["<error>text_requested</error>"]), "a sign") == ("", "text_requested")
    refusal = "I cannot help with that request because it names a real person and asks for something I will not write out in any detail at all for you today."
    assert upsampler.upsample_prompt_local(FakeClip([refusal]), "x") == ("", "illicit_content")
    assert upsampler.upsample_prompt_local(FakeClip(["Sure."]), "x") == ("", "illicit_content")


def test_local_upsampler_revises_once():
    clip = FakeClip(["<caption>too short</caption>", f"<caption>{CAPTION}</caption>"])
    assert upsampler.upsample_prompt_local(clip, "bread") == (CAPTION, None)
    assert "Revise the caption" in clip.prompts[1]
    assert clip.prompts[0].startswith("<|im_start|>system\n") and clip.prompts[0].endswith("<think>\n\n</think>\n\n")


def test_merge_tool(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    a = {"model.language_model.layers.0.linear_attn.A_log": torch.arange(4, dtype=torch.float32), "lm_b": torch.ones(2, 3, dtype=torch.bfloat16)}
    b = {"model.visual.x": torch.arange(6, dtype=torch.bfloat16)}
    save_file(a, source / "s1.safetensors")
    save_file(b, source / "s2.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {**{k: "s1.safetensors" for k in a}, **{k: "s2.safetensors" for k in b}}}))
    output = tmp_path / "merged.safetensors"
    tool = HERE.parent / "tools/merge_qwen35.py"
    subprocess.run([sys.executable, str(tool), str(source), str(output)], check=True, capture_output=True)
    merged = load_file(output)
    assert set(merged) == set(a) | set(b)
    for key, value in {**a, **b}.items():
        assert merged[key].dtype == value.dtype and torch.equal(merged[key], value)
