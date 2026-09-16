import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import torch
from PIL import Image

MAX_PROMPT_LENGTH = 1000
MIN_NUM_IMAGES, MAX_NUM_IMAGES = 1, 4
MIN_STEPS, MAX_STEPS = 1, 150
MIN_GUIDANCE, MAX_GUIDANCE = 0.0, 20.0

PipelineCall = Callable[..., list[Image.Image]]


class InvalidGenerationInput(ValueError):
    """Raised when generate_image is called with invalid input parameters."""


class PipelineExecutionError(RuntimeError):
    """Raised when the underlying SDXL pipeline call fails or times out."""


@dataclass(frozen=True)
class GeneratedCandidate:
    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str


def generate_image(
    prompt: str,
    *,
    negative_prompt: str = "",
    seed: int | None = None,
    num_images: int = 2,
    guidance_scale: float = 7.5,
    steps: int = 30,
    pipeline_call: PipelineCall | None = None,
    seed_factory: Callable[[], int] | None = None,
    timeout_seconds: float = 120.0,
) -> list[GeneratedCandidate]:
    """Generate `num_images` SDXL candidates for a prompt, one round of a session."""
    _validate_inputs(prompt, negative_prompt, num_images, guidance_scale, steps)

    resolved_seed = seed if seed is not None else (seed_factory or _random_seed)()
    if pipeline_call is None:
        _load_real_pipeline()  # one-time weight load; not counted against timeout_seconds
    call = pipeline_call or _default_pipeline_call

    images = _run_with_timeout(
        call,
        timeout_seconds,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images=num_images,
        guidance_scale=guidance_scale,
        steps=steps,
        seed=resolved_seed,
    )

    return [
        GeneratedCandidate(
            image=image,
            seed=resolved_seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        for image in images
    ]


def _validate_inputs(
    prompt: str,
    negative_prompt: str,
    num_images: int,
    guidance_scale: float,
    steps: int,
) -> None:
    if not prompt.strip():
        raise InvalidGenerationInput("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise InvalidGenerationInput(f"prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if len(negative_prompt) > MAX_PROMPT_LENGTH:
        raise InvalidGenerationInput(f"negative_prompt exceeds {MAX_PROMPT_LENGTH} characters")
    if not (MIN_NUM_IMAGES <= num_images <= MAX_NUM_IMAGES):
        raise InvalidGenerationInput(
            f"num_images must be between {MIN_NUM_IMAGES} and {MAX_NUM_IMAGES}"
        )
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        raise InvalidGenerationInput(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    if not (MIN_GUIDANCE <= guidance_scale <= MAX_GUIDANCE):
        raise InvalidGenerationInput(
            f"guidance_scale must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}"
        )


def _random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def _run_with_timeout(
    fn: PipelineCall, timeout_seconds: float, **kwargs: object
) -> list[Image.Image]:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        raise PipelineExecutionError(f"generation timed out after {timeout_seconds}s") from exc
    except Exception as exc:
        raise PipelineExecutionError(f"pipeline call failed: {exc}") from exc
    finally:
        executor.shutdown(wait=False)


_real_pipeline = None


def _load_real_pipeline():
    global _real_pipeline
    if _real_pipeline is None:
        from diffusers import StableDiffusionXLPipeline

        _real_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
    return _real_pipeline


def _default_pipeline_call(
    *,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> list[Image.Image]:
    pipe = _load_real_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_images_per_prompt=num_images,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images
