import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.generate import (
    GeneratedCandidate,
    InvalidGenerationInput,
    PipelineExecutionError,
    generate_image,
)

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

OUTPUT_DIR = Path("outputs")


def generate_image_tool(
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    num_images: int = 2,
    guidance_scale: float = 7.5,
    steps: int = 30,
) -> dict:
    """Generate SDXL candidate images for a brief.

    Produces `num_images` candidates from `prompt` at a shared seed (a
    fresh random seed is used when `seed` is omitted). Returns saved
    candidate image paths and metadata on success, or a structured error
    (`invalid_input` or `pipeline_failed`) on failure.
    """
    try:
        candidates = generate_image(
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_images=num_images,
            guidance_scale=guidance_scale,
            steps=steps,
        )
    except InvalidGenerationInput as exc:
        return {"ok": False, "error": "invalid_input", "message": str(exc)}
    except PipelineExecutionError as exc:
        return {"ok": False, "error": "pipeline_failed", "message": str(exc)}

    return {"ok": True, "candidates": [_save_candidate(c) for c in candidates]}


def _save_candidate(candidate: GeneratedCandidate) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
    candidate.image.save(path)
    return {
        "path": str(path),
        "seed": candidate.seed,
        "prompt": candidate.prompt,
        "negative_prompt": candidate.negative_prompt,
    }


def register(mcp: "MCPServer") -> None:
    mcp.tool()(generate_image_tool)
