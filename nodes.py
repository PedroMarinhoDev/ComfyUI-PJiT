"""P-JiT nodes. The model, the text encoder and the (pixel-space) VAE load separately and everything else is stock:

    Load P-JiT Model ─────────────────────────────┐
    Load P-JiT Text Encoder ─ CLIP Text Encode (x2) ─ KSampler ─ VAE Decode ─ Save Image
    P-JiT Empty Image ────────────────────────────┘      │
    Load VAE (pixel_space) ──────────────────────────────┘

Sampling contract (Linum's sampler): 50 steps, sampler `euler`, scheduler `simple`, cfg 15, an empty negative prompt.
The loader carries the guidance (APG) and the noise scale (4). P-JiT has no VAE: its LATENT is the image.
"""
import logging

import torch
from typing_extensions import override

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, IO

from .pjit import guidance, loader, noise, text, upsampler

CATEGORY = "P-JiT"

# The five ~1 MP sizes the model was trained at, (width, height), named as the reference's --aspect_ratio.
SIZES = {
    "square 1024x1024": (1024, 1024),
    "tall 864x1216": (864, 1216),
    "4:3 1152x864": (1152, 864),
    "3:2 1248x832": (1248, 832),
    "16:9 1280x720": (1280, 720),
}
CUSTOM_SIZE = "custom (width, height below)"
EXPLICIT_ERRORS = {
    upsampler.ERROR_ILLICIT_CONTENT: "the prompt names a real person or asks for explicit content",
    upsampler.ERROR_TEXT_REQUESTED: "the prompt asks for writing in the image, which this model cannot draw; describe the scene without the text",
}


def _default(options, needle):
    return next((name for name in options if needle in name.lower()), options[0] if options else None)


class PJiTLoadModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        return IO.Schema(
            node_id="PJiTLoadModel", display_name="Load P-JiT Model", category=CATEGORY,
            description="Loads the P-JiT DiT from diffusion_models as a native MODEL with dynamic VRAM: Linum's fp32 release, the bf16 file or the int8 ConvRot file. Reference guidance (APG) and the noise scale of 4 come with it, so the stock KSampler works with cfg 15.",
            inputs=[
                IO.Combo.Input("model_name", options=files, default=_default(files, "pjit")),
                IO.Combo.Input("weight_dtype", options=list(loader.WEIGHT_DTYPES), default="default",
                               tooltip="How the weights are held. default keeps the file's type. The model computes in bf16 either way, so bf16 gives the same pictures as fp32 at half the memory (the norms stay fp32). Ignored for the int8 file."),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(cls, model_name, weight_dtype) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return IO.NodeOutput(loader.load_dit(path, weight_dtype=weight_dtype))


class PJiTLoadTextEncoder(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.get_filename_list("text_encoders")
        return IO.Schema(
            node_id="PJiTLoadTextEncoder", display_name="Load P-JiT Text Encoder", category=CATEGORY,
            description="Loads Qwen3.5-4B from text_encoders (bf16, or the smaller int8 ConvRot, W6A8 and W4A8 files) as a CLIP that outputs what P-JiT was trained on: blocks 7, 15 and 27 side by side, chat template with thinking off, at most 256 tokens. Works with CLIP Text Encode and Generate Text.",
            inputs=[
                IO.Combo.Input("text_encoder", options=files, default=_default(files, "qwen3.5_4b")),
                IO.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True),
            ],
            outputs=[IO.Clip.Output()],
        )

    @classmethod
    def execute(cls, text_encoder, device) -> IO.NodeOutput:
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        return IO.NodeOutput(text.load_text_encoder(path, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options))


class PJiTGuidance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PJiTGuidance", display_name="P-JiT Guidance (APG)", category=CATEGORY,
            description="Changes the adaptive projected guidance the loader sets. The sampler's cfg is the guidance scale, applied as v_cond + cfg * guidance, so the release setting is 15. Only needed to change the settings below or turn it off.",
            inputs=[
                IO.Model.Input("model"),
                IO.Boolean.Input("enabled", default=True, tooltip="Off is plain CFG, which burns out at the release's cfg 15."),
                IO.Float.Input("momentum", default=-0.75, min=-5.0, max=1.0, step=0.01, tooltip="Running average of the guidance across steps. Negative values damp oscillation."),
                IO.Float.Input("eta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Weight of the guidance component parallel to the conditional velocity."),
                IO.Float.Input("rescale", default=10.0, min=0.0, max=1000.0, step=0.1, tooltip="Largest per-image L2 norm of the velocity difference, 0 for no limit."),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, momentum, eta, rescale) -> IO.NodeOutput:
        model = model.clone()
        if enabled:
            model.set_model_sampler_cfg_function(guidance.make_apg_function(momentum, eta, rescale))
        else:
            model.model_options.pop("sampler_cfg_function", None)
        return IO.NodeOutput(model)


class PJiTEmptyImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PJiTEmptyImage", display_name="P-JiT Empty Image", category=CATEGORY,
            description="An empty image-sized LATENT. P-JiT works on pixels, and was trained only at the five sizes listed.",
            inputs=[
                IO.Combo.Input("size", options=list(SIZES) + [CUSTOM_SIZE], default="square 1024x1024"),
                IO.Int.Input("width", default=1024, min=32, max=16384, step=8, tooltip="Used with the custom size. Sides that are not a multiple of 32 are padded and cropped back."),
                IO.Int.Input("height", default=1024, min=32, max=16384, step=8, tooltip="Used with the custom size."),
                IO.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, size, width, height, batch_size) -> IO.NodeOutput:
        width, height = SIZES.get(size, (width, height))
        samples = torch.zeros((batch_size, 3, height, width), device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples": samples})


class PJiTScheduler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PJiTScheduler", display_name="P-JiT Scheduler", category=CATEGORY,
            description="The reference's exact schedule, torch.linspace(1, 0, steps + 1). The stock simple scheduler differs in the last bit at a few steps and is slightly uneven for step counts that do not divide 1000, which barely shows.",
            inputs=[
                IO.Int.Input("steps", default=50, min=1, max=1000),
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[IO.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, steps, denoise) -> IO.NodeOutput:
        if denoise <= 0.0:
            return IO.NodeOutput(torch.FloatTensor([]))
        total = int(steps / denoise)
        return IO.NodeOutput(torch.linspace(1.0, 0.0, total + 1)[-(steps + 1):])


class PJiTH100Noise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PJiTH100Noise", display_name="P-JiT Noise (H100)", category=CATEGORY,
            description="For SamplerCustomAdvanced: noise drawn exactly as an H100 SXM draws it, so a seed gives Linum's published image on any NVIDIA GPU. Compiles a small Triton kernel on first use. Not needed otherwise: stock noise is just as good, it only gives other pictures.",
            inputs=[IO.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True)],
            outputs=[IO.Noise.Output()],
        )

    @classmethod
    def execute(cls, noise_seed) -> IO.NodeOutput:
        return IO.NodeOutput(noise.H100Noise(noise_seed))


class PJiTPromptUpsampler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PJiTPromptUpsampler", display_name="P-JiT Prompt Upsampler", category=CATEGORY,
            description="Rewrites a short idea into the dense 50-80 word caption P-JiT was trained on, with Linum's system prompt and checks. Local runs Qwen3.5-4B through the connected text encoder. OpenRouter sends the prompt to Claude Opus 5.5 at openrouter.ai, as the reference does, and reads the key from the OPENROUTER_API_KEY environment variable.",
            inputs=[
                IO.String.Input("prompt", multiline=True, default="oil painting of weathered fisherman smoking a pipe"),
                IO.Combo.Input("backend", options=["local", "openrouter"], default="local"),
                IO.Combo.Input("size", options=list(SIZES) + [CUSTOM_SIZE], default="square 1024x1024", tooltip="The frame the caption is composed for."),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=IO.ControlAfterGenerate.fixed, tooltip="Local backend only. Fixed by default, so changing the image seed doesn't rewrite the caption."),
                IO.Clip.Input("clip", optional=True, tooltip="The text encoder, for the local backend."),
            ],
            outputs=[IO.String.Output(display_name="caption")],
        )

    @classmethod
    def execute(cls, prompt, backend, size, seed, clip=None) -> IO.NodeOutput:
        width, height = SIZES.get(size, (1024, 1024))
        if backend == "openrouter":
            result = upsampler.upsample_prompt(prompt, height=height, width=width)
            caption, error = result.upsampled_prompt, result.error
        else:
            if clip is None:
                raise ValueError("the local upsampler needs the text encoder: connect Load P-JiT Text Encoder to clip")
            caption, error = upsampler.upsample_prompt_local(clip, prompt, height=height, width=width, seed=seed)
        if error:
            raise ValueError(f"P-JiT upsampler: {error} ({EXPLICIT_ERRORS[error]})")
        logging.info("P-JiT upsampled caption: %s", caption)
        return IO.NodeOutput(caption)


class PJiTExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [PJiTLoadModel, PJiTLoadTextEncoder, PJiTGuidance, PJiTEmptyImage, PJiTScheduler, PJiTH100Noise, PJiTPromptUpsampler]


async def comfy_entrypoint() -> PJiTExtension:
    return PJiTExtension()
