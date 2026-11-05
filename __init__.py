"""P-JiT 21K (Linum-AI/pjit-21k) as native ComfyUI nodes: a pixel-space text-to-image model, with no VAE."""
if __package__:  # ComfyUI imports the pack as a package; pytest may import this file on its own
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
