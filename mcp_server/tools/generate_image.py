from typing import TYPE_CHECKING

from PIL import Image as PILImage

from mcp_server.session import ITERATION_CAP, NoActiveSession, session
from mcp_server.tools._media import save_image, score_or_none, thumbnail
from pipeline.generate import (
    GeneratedCandidate,
    InvalidGenerationInput,
    PipelineExecutionError,
    generate_image,
)

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def generate_image_tool(
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    num_images: int = 2,
    guidance_scale: float = 7.5,
    steps: int = 30,
    continue_session: bool = False,
) -> list | dict:
    """Generate SDXL candidate images for a brief, one round of a session.

    `continue_session=False` (the default) starts a fresh session at
    `prompt` — use this for a new brief. `continue_session=True` continues
    the current session for a fix round (reprompt or a param retry); it
    fails with `no_active_session` if no session was started first.

    A session gets 5 rounds total (ADR 0003), shared across every kind of
    fix. Once the cap is reached, further `continue_session=True` calls
    don't run generation again — they return the best-scoring round's
    candidates again with `cap_hit: true`, never silently as a fresh
    success and never discarding the work already done.

    Each candidate is automatically scored against the session's brief
    with CLIP (`clip_score`, 0-100, `null` if scoring failed) — advisory
    only (ADR 0004), the agent's own vision judgment is still primary.

    On success, returns a list mixing viewable JPEG thumbnails (resized;
    the full-resolution PNG is only on disk, at each candidate's `path`)
    with a trailing metadata dict (`ok`, `cap_hit`, `rounds_used`,
    `candidates`). On failure, returns a single structured error dict
    (`invalid_input` or `pipeline_failed` before generation runs;
    `no_active_session` if `continue_session=True` with nothing started).

    The first call in a session is slower than later ones (one-time SDXL
    weight load, several GB). A `pipeline_failed` result from a timeout
    does not guarantee GPU work actually stopped — the underlying call is
    abandoned in the background, not cancelled.
    """
    if continue_session:
        try:
            if session.cap_reached():
                return _cap_hit_result()
        except NoActiveSession as exc:
            return {"ok": False, "error": "no_active_session", "message": str(exc)}
    else:
        session.start_new(prompt)

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

    brief = session.current().brief
    saved = [_save_candidate(c, brief) for c in candidates]
    session.record_round(saved)

    return [
        *(thumbnail(c.image) for c in candidates),
        {
            "ok": True,
            "cap_hit": False,
            "rounds_used": session.current().rounds_used,
            "candidates": saved,
        },
    ]


def _cap_hit_result() -> list:
    best = session.best_scoring_round()
    return [
        *(thumbnail(PILImage.open(c["path"])) for c in best),
        {
            "ok": True,
            "cap_hit": True,
            "rounds_used": session.current().rounds_used,
            "candidates": best,
            "message": f"iteration cap ({ITERATION_CAP}) reached; returning the best-scoring round",
        },
    ]


def _save_candidate(candidate: GeneratedCandidate, brief: str) -> dict:
    path = save_image(candidate.image)
    return {
        "path": str(path),
        "seed": candidate.seed,
        "prompt": candidate.prompt,
        "negative_prompt": candidate.negative_prompt,
        "clip_score": score_or_none(candidate.image, brief),
    }


def register(mcp: "MCPServer") -> None:
    mcp.tool(structured_output=False)(generate_image_tool)
