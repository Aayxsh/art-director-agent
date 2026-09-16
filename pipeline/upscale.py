import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image

from pipeline._runtime import PipelineExecutionError, random_seed, run_with_timeout
from pipeline.generate import MAX_PROMPT_LENGTH

MIN_SCALE, MAX_SCALE = 1.0, 4.0
MIN_STEPS, MAX_STEPS = 1, 150
MIN_GUIDANCE, MAX_GUIDANCE = 0.0, 20.0
DEFAULT_SCALE = 2.0
DEFAULT_STEPS = 15  # refiner detail pass, not a fresh generation — fewer steps needed

PipelineCall = Callable[..., Image.Image]

__all__ = ["InvalidUpscaleInput", "PipelineExecutionError", "UpscaledCandidate", "upscale"]


class InvalidUpscaleInput(ValueError):
    """Raised when upscale is called with invalid input parameters."""


@dataclass(frozen=True)
class UpscaledCandidate:
    image: Image.Image
    source_path: str
    scale: float


def upscale(
    source_path: str,
    *,
    prompt: str,
    negative_prompt: str = "",
    scale: float = DEFAULT_SCALE,
    guidance_scale: float = 7.5,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
    seed_factory: Callable[[], int] | None = None,
    pipeline_call: PipelineCall | None = None,
    timeout_seconds: float = 180.0,
) -> UpscaledCandidate:
    """Upscale the accepted candidate at `source_path` (ADR 0009): an SDXL
    refiner pass at native resolution (detail, not pixel dimensions) followed
    by a plain Lanczos resize to `scale`x — no dedicated super-resolution
    model, no extra VRAM beyond what ADR 0002 already budgeted.

    Not part of the critique loop's round cap (ADR 0003) — this runs once,
    after a candidate is Acceptable or at cap-out, never mid-loop.
    """
    _validate_inputs(source_path, prompt, negative_prompt, scale, guidance_scale, steps)

    source_image = Image.open(source_path).convert("RGB")
    resolved_seed = seed if seed is not None else (seed_factory or random_seed)()
    call = pipeline_call or _default_pipeline_call
    if call is _real_pipeline_call:
        _load_real_refiner_pipeline()  # one-time weight load; not counted against timeout_seconds

    refined = run_with_timeout(
        call,
        timeout_seconds,
        image=source_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        steps=steps,
        seed=resolved_seed,
    )

    target_size = (round(refined.width * scale), round(refined.height * scale))
    resized = refined.resize(target_size, Image.LANCZOS)

    return UpscaledCandidate(image=resized, source_path=source_path, scale=scale)


def _validate_inputs(
    source_path: str,
    prompt: str,
    negative_prompt: str,
    scale: float,
    guidance_scale: float,
    steps: int,
) -> None:
    if not os.path.isfile(source_path):
        raise InvalidUpscaleInput(f"source_path does not exist: {source_path}")
    if not prompt.strip():
        raise InvalidUpscaleInput("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise InvalidUpscaleInput(f"prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if len(negative_prompt) > MAX_PROMPT_LENGTH:
        raise InvalidUpscaleInput(f"negative_prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        raise InvalidUpscaleInput(f"scale must be between {MIN_SCALE} and {MAX_SCALE}")
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        raise InvalidUpscaleInput(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    if not (MIN_GUIDANCE <= guidance_scale <= MAX_GUIDANCE):
        raise InvalidUpscaleInput(
            f"guidance_scale must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}"
        )


_real_refiner_pipeline = None


def _load_real_refiner_pipeline():
    global _real_refiner_pipeline
    if _real_refiner_pipeline is None:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        _real_refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
    return _real_refiner_pipeline


def _real_pipeline_call(
    *,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    pipe = _load_real_refiner_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images[0]


# Sanctioned test seam (see pipeline/generate.py's _default_pipeline_call for
# the full rationale) — monkeypatch this name in MCP-tool-layer tests.
_default_pipeline_call = _real_pipeline_call
