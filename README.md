<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png">
  <img alt="ComfyUI-PJiT: Linum's P-JiT 21K natively in ComfyUI" src="assets/banner-light.png" width="100%">
</picture>

<p align="center">
  <a href="#requirements"><img alt="ComfyUI v0.44+" src="https://img.shields.io/badge/ComfyUI-v0.44%2B-F2A65A?style=flat-square&labelColor=555"></a>
  <a href="#model-files"><img alt="VRAM from 4 GB" src="https://img.shields.io/badge/VRAM-from%204%20GB-F2A65A?style=flat-square&labelColor=555"></a>
  <a href="#quick-start"><img alt="No pip install" src="https://img.shields.io/badge/pip%20install-none-F2A65A?style=flat-square&labelColor=555"></a>
  <a href="#license-and-credits"><img alt="Code GPL-3.0, weights Apache-2.0" src="https://img.shields.io/badge/license-GPL--3.0%20%C2%B7%20weights%20Apache--2.0-F2A65A?style=flat-square&labelColor=555"></a>
</p>

<p align="center">
  <a href="#quick-start"><img alt="Get started" src="assets/button-start.svg" height="40"></a>
  <a href="https://huggingface.co/PedroMarinhoDev/P-JiT-21K-ComfyUI"><img alt="Weights on Hugging Face" src="assets/button-weights.svg" height="40"></a>
  <a href="example_workflows"><img alt="Workflows" src="assets/button-workflow.svg" height="40"></a>
  <a href="https://huggingface.co/Linum-AI/pjit-21k"><img alt="P-JiT by Linum" src="assets/button-linum.svg" height="40"></a>
</p>

Run **[P-JiT 21K](https://huggingface.co/Linum-AI/pjit-21k)** by Linum **natively in ComfyUI**. P-JiT is a 2B-parameter
text-to-image model that draws directly in pixels: there is no VAE and no latent space. Linum trained it for about $21K
of compute and calls it a research checkpoint with rough edges. Here it is a real `MODEL` on ComfyUI's memory manager,
its Qwen3.5-4B text encoder a real `CLIP`, and the stock KSampler drives it. Nothing in ComfyUI is changed.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/showcase-dark.jpg">
  <img alt="Seven P-JiT images: a panda in a bamboo grove, a Scandinavian reading nook, a bowl of ramen, a Mediterranean street after rain, an elderly woman in a headscarf, a lighthouse at sunset and a hot-air balloon over fields at sunrise" src="assets/showcase-light.jpg" width="100%">
</picture>

<sub><code>pjit_21k_bf16</code> · 50 steps, euler, cfg 15 · RTX 3090 · nook, ramen, street, lighthouse and balloon: our captions, seed 42 · panda and portrait: Linum's sample captions with their seeds and noise</sub>

<table>
  <tr>
    <td width="33%" valign="top"><img src="assets/icons/native.svg" width="24" alt=""><br><b>ComfyUI's own nodes</b><br>Stock KSampler, CLIP Text Encode, Load VAE (<code>pixel_space</code>) and memory management. Guidance comes with the model.</td>
    <td width="33%" valign="top"><img src="assets/icons/pixels.svg" width="24" alt=""><br><b>Pixels, no VAE</b><br>The latent is the image, so live previews are the picture itself and image to image needs no encoder.</td>
    <td width="33%" valign="top"><img src="assets/icons/files.svg" width="24" alt=""><br><b>bf16 at half the size</b><br>The bf16 file gives the same pixels as Linum's fp32 release, at 4.4 GB instead of 8.7.</td>
  </tr>
  <tr>
    <td valign="top"><img src="assets/icons/speed.svg" width="24" alt=""><br><b>int8 ConvRot</b><br>2.3 GB and about 1.8x faster on ComfyUI's own int8 kernels, with small changes in fine detail.</td>
    <td valign="top"><img src="assets/icons/refine.svg" width="24" alt=""><br><b>A refiner pass</b><br>Image to image at low denoise, with Qwen3.5-4B describing the input: a second pass for images from other models.</td>
    <td valign="top"><img src="assets/icons/caption.svg" width="24" alt=""><br><b>Linum's samples, seed for seed</b><br>Their noise and schedule reproduce the published images, and the prompt upsampler writes captions from a short idea.</td>
  </tr>
</table>

## Quick start

1. **Install the nodes** and restart ComfyUI. Clone this repository into `custom_nodes`. There's nothing to
   `pip install`.

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PedroMarinhoDev/ComfyUI-PJiT
   ```

2. **Download the files.** The model comes from
   [PedroMarinhoDev/P-JiT-21K-ComfyUI](https://huggingface.co/PedroMarinhoDev/P-JiT-21K-ComfyUI), the text encoder
   from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5). There is no VAE file.

   | File | Folder | Size |
   |---|---|---|
   | [pjit_21k_bf16.safetensors](https://huggingface.co/PedroMarinhoDev/P-JiT-21K-ComfyUI/blob/main/diffusion_models/pjit_21k_bf16.safetensors) | `models/diffusion_models` | 4.37 GB |
   | [qwen3.5_4b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_bf16.safetensors) | `models/text_encoders` | 9.32 GB |

   ```bash
   cd ComfyUI/models
   hf download PedroMarinhoDev/P-JiT-21K-ComfyUI diffusion_models/pjit_21k_bf16.safetensors --local-dir .
   hf download Comfy-Org/Qwen3.5 text_encoders/qwen3.5_4b_bf16.safetensors --local-dir .
   ```

   The workflows list both files with their links, so ComfyUI offers to download what's missing when you open one.

3. **Open a workflow and run.** In ComfyUI, **Templates → ComfyUI-PJiT → pjit_text_to_image** (or drag
   [`example_workflows/pjit_text_to_image.json`](example_workflows/pjit_text_to_image.json) in).

**Linum's settings are already in the workflows:** 50 steps, euler, scheduler simple, **cfg 15** and an empty negative
prompt. **Load P-JiT Model** brings P-JiT's guidance (APG) and its noise scale with it, so the stock KSampler needs no
extra nodes. The model was trained at five sizes, which **P-JiT Empty Image** lists: 1024×1024, 864×1216, 1152×864,
1248×832 and 1280×720.

## Prompts

P-JiT was trained on dense captions of **50 to 80 words** saying what is visible, where it sits in the frame and how
it is lit. Short prompts give noticeably worse images: write captions like the ones in the workflows, or let
**P-JiT Prompt Upsampler** write one from a short idea.

It cannot draw text, signs, logos or posters, because Linum filtered those out of the training data. It also struggles
with full-body shots, crowds, several people interacting, and world knowledge such as the strings on a guitar.

## Requirements

| | Tested |
|---|---|
| ComfyUI | 0.44.0 (a clean release install, with the comfy-kitchen it pins) and master of 2026-11-05 |
| GPU | RTX 3090, 24 GB. Attention: pytorch, sage and xformers |
| VRAM | from 4 GB free with dynamic VRAM (below), plus room for one 1.27 GB buffer of the text encoder |
| Python packages | none beyond ComfyUI. P-JiT Noise (H100) compiles a Triton kernel on first use |

GPUs without bf16 (GTX 10 and RTX 20 series) and the CPU compute in fp32 instead, which ComfyUI decides as for its own
models; `--force-fp32` does the same. That path was checked on the RTX 3090 and on the CPU with a small model, not on
those GPUs.

## Model files

All three load in **Load P-JiT Model**. The model computes in bf16 whatever the file, as Linum's code does under
autocast, so the file only decides what is stored.

| File | Size | RTX 3090, 1024×1024 | Against Linum's images (portrait, birthday, panda) |
|---|---|---|---|
| **[pjit_21k_bf16](https://huggingface.co/PedroMarinhoDev/P-JiT-21K-ComfyUI/blob/main/diffusion_models/pjit_21k_bf16.safetensors)** | 4.37 GB | 0.34 s/step | 40.2, 39.5, 47.7 dB: the same pixels as fp32 |
| [pjit_21k_int8_convrot](https://huggingface.co/PedroMarinhoDev/P-JiT-21K-ComfyUI/blob/main/diffusion_models/pjit_21k_int8_convrot.safetensors) | 2.33 GB | 0.19 s/step | 32.8, 31.7, 40.3 dB: the same pictures, other fine detail |
| [Linum's fp32 release](https://huggingface.co/Linum-AI/pjit-21k/blob/main/model.safetensors) | 8.74 GB | 0.35 s/step | 40.2, 39.5, 47.7 dB |

- **bf16 is lossless here.** It keeps the norm gains and the output modulation in fp32 and rounds only the Linears and
  convs, which bf16 autocast rounds the same way before every matmul. The three test images came out pixel for pixel
  equal to the fp32 file's.
- **int8 ConvRot** stores q, k, v, o and the feed-forward of every block in ComfyUI's `int8_tensorwise` format with
  ConvRot (a Hadamard rotation before rounding), and the rest as the bf16 file does. The composition, faces and
  lighting stay; texture, candle positions and wrinkles change.
  On GPUs without int8 kernels ComfyUI dequantizes it, which saves memory but not time.
- Linum's own code takes 0.38 s/step on this card, and Linum's own code against itself, rerun in another batch shape,
  differs by 40.5 to 49.3 dB.

<details>
<summary><b>Speed with less free VRAM</b></summary>

RTX 3090 at 1024×1024, 50 steps, with another process holding the rest of the card. Dynamic VRAM streams what doesn't
fit, and the images are the same at every budget.

| Free VRAM at start | `pjit_21k_bf16` | `pjit_21k_int8_convrot` | fp32 release |
|---|---|---|---|
| all 24 GB | 0.34 s/step | 0.19 s/step | 0.35 s/step |
| 4 GiB | 0.39 s/step | 0.19 s/step | 0.97 s/step |

A whole 1024×1024 image takes about 17 s with bf16 and 10 s with int8, the caption included. The text encoder needs
room for a single 1.27 GB buffer: if other programs have filled the card, encoding stops with `VRAM grow failed`. Set
**Load P-JiT Text Encoder**'s `device` to `cpu` (fp32 there, about 2 minutes for a caption and a negative on an
i9-14900K), or free the memory.
</details>

## Workflows

Four, in [`example_workflows`](example_workflows) and under **Templates → ComfyUI-PJiT**:

| Workflow | |
|---|---|
| **pjit_text_to_image** | the plain graph, with stock nodes wherever there is one |
| **pjit_prompt_upsampler** | a short idea becomes a caption, then an image |
| **pjit_refiner** | image to image as a second pass, below |
| **pjit_linum_samples** | Linum's published samples, seed for seed, on any NVIDIA GPU |

### P-JiT as a refiner

**pjit_refiner** takes an image from any model, has the Qwen3.5-4B text encoder describe it in the caption style P-JiT
was trained on (stock **Generate Text** with the image), and redraws it at low denoise. Copy
[`example_inputs/pjit_refiner_example.png`](example_inputs) to `ComfyUI/input` for the softened FLUX.2 portrait it
opens with.

<img alt="The softened input on the left; P-JiT's pass at denoise 0.15 on the right, with skin texture added" src="example_workflows/pjit_refiner.jpg" width="60%">

**Denoise runs lower than in other models.** P-JiT's noise is 4× the image's scale (x_t = (1 − t)·image + 4t·noise), so
the same denoise removes more of the image: 0.12 here leaves what 0.35 leaves elsewhere. On that portrait **0.05**
smoothed it further, **0.1 to 0.2** added skin texture and fine detail while keeping the face, and **0.3** began to
change the face. The workflow uses 0.15 with 20 steps, which keeps the step size near the 50-step schedule it was
trained on. The input is scaled to about 1 megapixel, keeping its shape; any size runs.

## The nodes

| Node | What it does |
|---|---|
| **Load P-JiT Model** | The DiT as a `MODEL` on dynamic VRAM, from any of the three files. `weight_dtype` bf16 stores the fp32 file in bf16 at load (the same pixels). |
| **Load P-JiT Text Encoder** | Qwen3.5-4B as a `CLIP` that outputs what P-JiT was trained on: blocks 7, 15 and 27 side by side, chat template with thinking off, at most 256 tokens. Works with stock **CLIP Text Encode** and **Generate Text**. |
| **P-JiT Empty Image** | An empty image-sized `LATENT`, with the five trained sizes as presets. |
| **P-JiT Prompt Upsampler** | Rewrites a short idea into a caption with Linum's brief and checks, below. |
| **P-JiT Guidance (APG)** | Changes or turns off the guidance the loader sets. Not needed for Linum's settings. |
| **P-JiT Scheduler** | Linum's exact schedule, for `SamplerCustomAdvanced`. |
| **P-JiT Noise (H100)** | Noise drawn as an H100 SXM draws it, so a seed gives Linum's picture on any NVIDIA GPU. |

The stock **APG** node is not a substitute for **P-JiT Guidance**: it works on x_0, while P-JiT's guidance runs on
velocities, so its norm clip and projection see different numbers.

### The prompt upsampler

`local` runs Qwen3.5-4B through the text encoder you connect: no key, and nothing leaves the machine. Its captions are
plainer and shorter than the reference's. `openrouter` is what Linum's code does: it sends your prompt to Claude Opus
5.5 at openrouter.ai and reads the key from the `OPENROUTER_API_KEY` environment variable (there is no key widget,
because widget values are saved inside workflows and image metadata). **That backend is Linum's code unchanged and has
not been run here.** As in the reference, prompts that name a real person, ask for explicit content or ask for writing
in the image stop with an error; a 4B model checks this less reliably than Opus.

<details>
<summary><b>Checked against Linum's code</b></summary>

Their code, run as released on the same RTX 3090, is the reference (commit 9540393).

| Check | Result |
|---|---|
| Token ids (template, empty prompt, truncation at 256) | identical |
| Initial noise and schedule | identical, bit for bit |
| Guidance step | the same up to fp32 rounding (5e-7) |
| Text embeddings | relative error 8e-3 to 2e-2; their own bf16 run differs from their fp32 run by as much |
| DiT, fed their exact inputs at each step | relative error 2.5e-3 to 1.3e-2; their code rerun row by row differs from itself by as much |
| Final images, same seed (portrait, birthday, panda) | 40.2, 39.5, 47.7 dB |
| Their code against itself, cond and uncond split | 46.6, 40.5, 49.3 dB |
| Their code against itself, unpadded as these nodes run | 46.5, 43.4, 49.2 dB |

<img alt="Linum's samples from their code (left of each pair) and from these nodes (right), same seeds" src="assets/reference_vs_native.jpg" width="100%">

So the pictures are the same, with small differences in fine detail: sampling at cfg 15 is chaotic, and rounding at
each step adds up over 50 steps. The nodes sit up to 6 dB further from the reference than the reference does from
itself on the portrait. Every part checked alone is at its noise floor, so no systematic error was found, but a small
one can't be ruled out. Linum's published images came from FlashAttention-3 on an H100, which neither path here uses.
</details>

<details>
<summary><b>Building the files yourself</b></summary>

```bash
hf download Linum-AI/pjit-21k model.safetensors --local-dir pjit-21k
python tools/convert_pjit.py pjit-21k/model.safetensors ComfyUI/models/diffusion_models
```

Run it with ComfyUI's Python, which has comfy-kitchen. It writes `pjit_21k_bf16.safetensors` and
`pjit_21k_int8_convrot.safetensors` in about 15 seconds on the CPU. The int8 weights are rotated and rounded from fp32,
in groups of 256 along the input where the width allows and 64 for the 2944-wide inputs (ComfyUI's Hadamard sizes are
powers of 4).

The text encoder from Comfy-Org/Qwen3.5 holds the same tensors, byte for byte, as Qwen's two shards of
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). To build it from those instead:

```bash
python tools/merge_qwen35.py path/to/Qwen3.5-4B ComfyUI/models/text_encoders/qwen3.5_4b_bf16.safetensors
```
</details>

## Troubleshooting

<details>
<summary><b>The images are bland, oddly composed or ignore the prompt</b></summary>

Write a caption of 50 to 80 words, or use the upsampler. Keep cfg at 15 with the empty negative, and one of the five
trained sizes: other sizes run, but framing can suffer, with picture-in-a-frame borders at 1000×600.
</details>

<details>
<summary><b><code>VRAM grow failed</code> while encoding the caption</b></summary>

The text encoder needs one 1.27 GB buffer at once. Close what else holds the card, or set **Load P-JiT Text
Encoder**'s `device` to `cpu`.
</details>

<details>
<summary><b>P-JiT Noise (H100) fails with a Triton error</b></summary>

It compiles a small Triton kernel, which needs Triton and an NVIDIA GPU (on Windows, the `triton-windows` package).
Only that node needs it: with the stock noise every other workflow runs, and gives other pictures for the same seed.
</details>

## License and credits

The code of this pack: GPL-3.0 (`LICENSE`). `pjit/noise.py` and `pjit/upsampler.py` are adapted from Linum's code and
stay under Apache-2.0 (see `NOTICE`). P-JiT 21K and its weights: [Linum](https://www.linum.ai), Apache-2.0
([weights](https://huggingface.co/Linum-AI/pjit-21k), [code](https://github.com/Linum-AI/pjit-21k),
[blog post](https://www.linum.ai/field-notes/pjit-21k)); the files on Hugging Face are those weights converted.
Qwen3.5-4B: Qwen, Apache-2.0. The refiner example is a FLUX.2 Klein image.

```bibtex
@online{chopra2026pjit21k,
  title = {Training a Text-to-Image Model for \$21K},
  author = {Chopra, Sahil and Chopra, Manu},
  year = {2026},
  url = {https://www.linum.ai/field-notes/pjit-21k}
}
```
