from typing import TYPE_CHECKING

from PIL import Image as PILImage

from mcp_server.session import ITERATION_CAP, NoActiveSession, session
from mcp_server.tools._media import save_image, score_or_none, thumbnail
from pipeline.inpaint import (
    InpaintedCandidate,
    InvalidInpaintInput,
    PipelineExecutionError,
    inpaint,
)

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def inpaint_tool(
    path: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    guidance_scale: float = 7.5,
    steps: int = 30,
) -> list | dict:
    """Fix a localized/region defect (e.g. malformed hands) in a candidate
    from the current session, by SDXL-inpainting the box (x0,y0,x1,y1)
    (normalized 0-1 coordinates) with `prompt` describing what should be
    there instead. `path` must be a candidate from the current session
    (a `path` returned by a previous generate_image or inpaint call) —
    fails with `unknown_candidate` otherwise.

    Always continues the current session — there's no scenario where
    inpaint legitimately starts a fresh one — and fails with
    `no_active_session` if nothing was started via generate_image first.
    Shares generate_image's 5-round cap (ADR 0003): once reached, returns
    the best-scoring round again unchanged, flagged `cap_hit: true`.

    The result is automatically CLIP-scored against the session's brief
    (`clip_score`, 0-100, `null` if scoring failed) — advisory only
    (ADR 0004). Same response shape as generate_image_tool: a list of
    viewable JPEG thumbnails plus a trailing metadata dict on success, a
    single structured error dict (`invalid_input`, `unknown_candidate`,
    `pipeline_failed`, `no_active_session`) on failure.
    """
    try:
        if session.cap_reached():
            return _cap_hit_result()
    except NoActiveSession as exc:
        return {"ok": False, "error": "no_active_session", "message": str(exc)}

    known_paths = {c["path"] for c in session.current().candidates}
    if path not in known_paths:
        return {
            "ok": False,
            "error": "unknown_candidate",
            "message": f"{path} is not a candidate from the current session",
        }

    try:
        candidate = inpaint(
            path,
            bbox=(x0, y0, x1, y1),
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            steps=steps,
        )
    except InvalidInpaintInput as exc:
        return {"ok": False, "error": "invalid_input", "message": str(exc)}
    except PipelineExecutionError as exc:
        return {"ok": False, "error": "pipeline_failed", "message": str(exc)}

    saved = _save_candidate(candidate, session.current().brief)
    session.record_round([saved])

    return [
        thumbnail(candidate.image),
        {
            "ok": True,
            "cap_hit": False,
            "rounds_used": session.current().rounds_used,
            "candidates": [saved],
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


def _save_candidate(candidate: InpaintedCandidate, brief: str) -> dict:
    path = save_image(candidate.image)
    return {
        "path": str(path),
        "seed": candidate.seed,
        "prompt": candidate.prompt,
        "negative_prompt": candidate.negative_prompt,
        "source_path": candidate.source_path,
        "bbox": candidate.bbox,
        "clip_score": score_or_none(candidate.image, brief),
    }


def register(mcp: "MCPServer") -> None:
    mcp.tool(structured_output=False)(inpaint_tool)
