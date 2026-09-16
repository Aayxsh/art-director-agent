from typing import TYPE_CHECKING

from mcp_server.tools._media import save_image, thumbnail
from pipeline.upscale import (
    DEFAULT_SCALE,
    DEFAULT_STEPS,
    InvalidUpscaleInput,
    PipelineExecutionError,
    upscale,
)

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def upscale_tool(
    path: str,
    prompt: str,
    negative_prompt: str = "",
    scale: float = DEFAULT_SCALE,
    guidance_scale: float = 7.5,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
) -> list | dict:
    """Upscale an accepted candidate: an SDXL refiner detail pass plus a
    resize to `scale`x the original dimensions (ADR 0009). `prompt` should
    be the same prompt used to generate `path`, for refiner quality.

    Not part of the critique loop and doesn't touch the session round cap
    (ADR 0003) — call this once a candidate is Acceptable (or at cap-out),
    not mid-loop. No active session required.

    Returns a list of one viewable JPEG thumbnail plus a trailing metadata
    dict (`ok`, `path`, `source_path`, `scale`) on success, or a single
    structured error dict (`invalid_input`, `pipeline_failed`) on failure.
    """
    try:
        candidate = upscale(
            path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=scale,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )
    except InvalidUpscaleInput as exc:
        return {"ok": False, "error": "invalid_input", "message": str(exc)}
    except PipelineExecutionError as exc:
        return {"ok": False, "error": "pipeline_failed", "message": str(exc)}

    saved_path = save_image(candidate.image)

    return [
        thumbnail(candidate.image),
        {
            "ok": True,
            "path": str(saved_path),
            "source_path": candidate.source_path,
            "scale": candidate.scale,
        },
    ]


def register(mcp: "MCPServer") -> None:
    mcp.tool(structured_output=False)(upscale_tool)
