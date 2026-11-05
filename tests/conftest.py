"""Loads the pack by path under a package name, so its nodes.py doesn't collide with ComfyUI's.

    PYTHONPATH=~/AI/ComfyUI ~/AI/ComfyUI/.venv/bin/python -m pytest tests

CPU only and no weights needed. The comparisons against Linum's code skip unless their repository is at PJIT_REFERENCE
(default ~/AI/research/pjit-21k-src); the checkpoint test skips unless the file is at PJIT_DIT.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from comfy.cli_args import args  # noqa: E402

args.cpu = True  # before comfy.model_management is imported
PACK = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get("PJIT_REFERENCE", Path.home() / "AI/research/pjit-21k-src"))
DIT_FILE = Path(os.environ.get("PJIT_DIT", Path.home() / "AI/ComfyUI/models/diffusion_models/pjit_21k_fp32.safetensors"))

spec = importlib.util.spec_from_file_location("comfyui_pjit", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
package = importlib.util.module_from_spec(spec)
sys.modules["comfyui_pjit"] = package
spec.loader.exec_module(package)


@pytest.fixture(scope="session")
def reference():
    if not (REFERENCE / "pjit_21k/model.py").exists():
        pytest.skip("Linum's repository is not available")
    sys.path.insert(0, str(REFERENCE))
    return REFERENCE


@pytest.fixture(scope="session")
def dit_file():
    if not DIT_FILE.exists():
        pytest.skip("the P-JiT checkpoint is not available")
    return DIT_FILE
