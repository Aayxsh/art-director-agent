from collections.abc import Callable

import torch
from PIL import Image

from pipeline._runtime import PipelineExecutionError, run_with_timeout

ScoreCall = Callable[..., float]

__all__ = ["PipelineExecutionError", "score_image"]


def score_image(
    image: Image.Image,
    prompt: str,
    *,
    pipeline_call: ScoreCall | None = None,
    timeout_seconds: float = 30.0,
) -> float:
    """CLIP prompt-image alignment score (0-100), advisory only (ADR 0004).

    Not a hard gate — the agent's own vision critique is still the primary
    judge of "good enough." The one-time CLIP model load is not counted
    against `timeout_seconds`, matching every other pipeline operation.
    """
    call = pipeline_call or _default_pipeline_call
    if call is _real_pipeline_call:
        _load_real_clip()  # one-time weight load; not counted against timeout_seconds

    return run_with_timeout(call, timeout_seconds, image=image, prompt=prompt)


_real_clip_model = None
_real_clip_processor = None


def _load_real_clip():
    global _real_clip_model, _real_clip_processor
    if _real_clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        checkpoint = "openai/clip-vit-large-patch14"
        _real_clip_model = CLIPModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to(
            "cuda"
        )
        _real_clip_processor = CLIPProcessor.from_pretrained(checkpoint)
    return _real_clip_model, _real_clip_processor


def _real_pipeline_call(*, image: Image.Image, prompt: str) -> float:
    model, processor = _load_real_clip()
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {
        k: (v.to("cuda", dtype=torch.float16) if torch.is_floating_point(v) else v.to("cuda"))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    similarity = (image_embeds @ text_embeds.T).item()
    return max(similarity, 0.0) * 100


# Sanctioned test seam (see pipeline/generate.py's _default_pipeline_call for
# the full rationale) — monkeypatch this name in MCP-tool-layer tests.
_default_pipeline_call = _real_pipeline_call
