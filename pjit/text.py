"""The caption encoder: Qwen3.5-4B through ComfyUI's own implementation (`comfy/text_encoders/qwen35.py`).

The reference (`pjit_21k/text_encoder.py`) applies the chat template with thinking disabled, truncates to 256 tokens and
concatenates the outputs of blocks 7, 15 and 27 along the feature axis: 3 x 2560 = 7680 per token (the FLUX.2 recipe).

ComfyUI's stock tokenizer builds the same template. Its `intermediate_output` tap reads the tokens *before* a layer, so
the output of block i is index i + 1: [8, 16, 28], which are HF's `hidden_states[i + 1]`.
"""
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.text_encoders.hunyuan_video
import comfy.text_encoders.qwen35
import comfy.utils
from comfy import sd1_clip

MODEL_TYPE = "qwen35_4b"
MAX_TOKENS = 256
TAP_LAYERS = [8, 16, 28]  # the outputs of blocks 7, 15, 27


class PJiTTokenizer(comfy.text_encoders.qwen35.Qwen35ImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=MODEL_TYPE)

    def tokenize_with_weights(self, text, return_word_ids=False, truncate=True, **kwargs):
        tokens = super().tokenize_with_weights(text, return_word_ids=return_word_ids, **kwargs)
        if truncate and "system_prompt" not in kwargs:
            # a caption past the limit loses its end and the assistant turn, as it does in the reference
            tokens[MODEL_TYPE] = [row[:MAX_TOKENS] for row in tokens[MODEL_TYPE][:1]]
        return tokens


class PJiTClipModel(comfy.text_encoders.qwen35.Qwen35ClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, mtp=False):
        super().__init__(device=device, layer=TAP_LAYERS, layer_idx=None, dtype=dtype, attention_mask=attention_mask,
                         model_options=model_options, model_type=MODEL_TYPE, mtp=mtp)


class PJiTTextEncoder(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, mtp=False):
        clip_model = lambda **kw: PJiTClipModel(**kw, mtp=mtp)
        super().__init__(device=device, dtype=dtype, name=MODEL_TYPE, clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        # [B, 3, L, 2560] -> [B, L, 7680], the block outputs side by side per token
        return out.movedim(1, 2).flatten(2), pooled, extra


def text_encoder_class(dtype_llama=None, llama_quantization_metadata=None, mtp=False):
    class PJiTTextEncoder_(PJiTTextEncoder):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = dict(model_options, quantization_metadata=llama_quantization_metadata)
            super().__init__(device=device, dtype=dtype, model_options=model_options, mtp=mtp)

    return PJiTTextEncoder_


class Target:
    params = {}
    clip = None
    tokenizer = PJiTTokenizer


def load_text_encoder(path, embedding_directory=None, model_options={}, disable_dynamic=False):
    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
    if "model.layers.0.linear_attn.A_log" not in state_dict:
        raise ValueError(f"{os.path.basename(path)} is not a Qwen3.5 text encoder (no linear_attn.A_log in layer 0)")
    width = state_dict["model.layers.0.input_layernorm.weight"].shape[0]
    if width != 2560:
        raise ValueError(f"P-JiT needs Qwen3.5-4B (hidden size 2560); {os.path.basename(path)} has hidden size {width}")

    target = Target()
    detected = comfy.text_encoders.hunyuan_video.llama_detect(state_dict)
    if model_options.get("load_device") == torch.device("cpu") and "llama_quantization_metadata" not in detected:
        # bf16 is emulated on most CPUs: 3.5 minutes for a caption and a negative on an i9-14900K
        detected["dtype_llama"] = torch.float32
    target.clip = text_encoder_class(**detected, mtp="mtp.fc.weight" in state_dict)
    parameters = comfy.utils.calculate_parameters(state_dict)
    clip = comfy.sd.CLIP(target, embedding_directory=embedding_directory, parameters=parameters, state_dict=[state_dict],
                         model_options=model_options, disable_dynamic=disable_dynamic)
    logging.info("P-JiT: loaded text encoder %s, tapping block outputs 7, 15, 27", os.path.basename(path))
    return clip
