import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image

from pipeline._runtime import PipelineExecutionError, random_seed, run_with_timeout
from pipeline.generate import MAX_PROMPT_LENGTH, _load_real_pipeline

MIN_STEPS, MAX_STEPS = 1, 150
MIN_GUIDANCE, MAX_GUIDANCE = 0.0, 20.0

Bbox = tuple[float, float, float, float]
PipelineCall = Callable[..., Image.Image]

__all__ = ["InpaintedCandidate", "InvalidInpaintInput", "PipelineExecutionError", "inpaint"]


class InvalidInpaintInput(ValueError):
    """Raised when inpaint is called with invalid input parameters."""


@dataclass(frozen=True)
class InpaintedCandidate:
    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str
    source_path: str
    bbox: Bbox


def inpaint(
    source_path: str,
    *,
    bbox: Bbox,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    guidance_scale: float = 7.5,
    steps: int = 30,
    pipeline_call: PipelineCall | None = None,
    seed_factory: Callable[[], int] | None = None,
    timeout_seconds: float = 120.0,
) -> InpaintedCandidate:
    """Inpaint the region of `source_path` given by `bbox` (normalized x0,y0,x1,y1).

    Reuses the already-loaded base SDXL pipeline's components (ADR 0008) —
    no separate inpainting checkpoint, no extra VRAM. The one-time base
    pipeline weight load is not counted against `timeout_seconds`, matching
    generate_image (pipeline/generate.py).
    """
    _validate_inputs(source_path, bbox, prompt, negative_prompt, guidance_scale, steps)

    source_image = Image.open(source_path).convert("RGB")
    mask = _build_mask(source_image.size, bbox)
    resolved_seed = seed if seed is not None else (seed_factory or random_seed)()
    call = pipeline_call or _default_pipeline_call
    if call is _real_pipeline_call:
        _load_real_pipeline()  # base pipeline warm-up; not counted against timeout_seconds

    image = run_with_timeout(
        call,
        timeout_seconds,
        image=source_image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        steps=steps,
        seed=resolved_seed,
    )

    return InpaintedCandidate(
        image=image,
        seed=resolved_seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_path=source_path,
        bbox=bbox,
    )


def _validate_inputs(
    source_path: str,
    bbox: Bbox,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    steps: int,
) -> None:
    if not os.path.isfile(source_path):
        raise InvalidInpaintInput(f"source_path does not exist: {source_path}")
    x0, y0, x1, y1 = bbox
    if not all(0.0 <= v <= 1.0 for v in bbox):
        raise InvalidInpaintInput("bbox values must be within [0, 1]")
    if x0 >= x1 or y0 >= y1:
        raise InvalidInpaintInput("bbox must satisfy x0 < x1 and y0 < y1")
    if not prompt.strip():
        raise InvalidInpaintInput("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise InvalidInpaintInput(f"prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if len(negative_prompt) > MAX_PROMPT_LENGTH:
        raise InvalidInpaintInput(f"negative_prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        raise InvalidInpaintInput(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    if not (MIN_GUIDANCE <= guidance_scale <= MAX_GUIDANCE):
        raise InvalidInpaintInput(
            f"guidance_scale must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}"
        )


def _build_mask(size: tuple[int, int], bbox: Bbox) -> Image.Image:
    width, height = size
    x0, y0, x1, y1 = bbox
    mask = Image.new("L", size, color=0)
    box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
    mask.paste(255, box)
    return mask


_real_inpaint_pipeline = None


def _load_real_inpaint_pipeline():
    global _real_inpaint_pipeline
    if _real_inpaint_pipeline is None:
        from diffusers import StableDiffusionXLInpaintPipeline

        base_pipe = _load_real_pipeline()
        _real_inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pipe(base_pipe)
    return _real_inpaint_pipeline


def _real_pipeline_call(
    *,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    pipe = _load_real_inpaint_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images[0]


# Sanctioned test seam (see pipeline/generate.py's _default_pipeline_call for
# the full rationale) — monkeypatch this name in MCP-tool-layer tests.
_default_pipeline_call = _real_pipeline_call
