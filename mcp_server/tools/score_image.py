from typing import TYPE_CHECKING

from PIL import Image as PILImage

from eval.scoring import PipelineExecutionError, score_image
from mcp_server.session import NoActiveSession, session

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def score_image_tool(path: str, prompt: str | None = None) -> dict:
    """Re-check a candidate's CLIP prompt-image alignment (0-100).

    `path` must be a candidate from the current session. `prompt`
    defaults to the session's brief (matching what generate_image/inpaint
    already score against automatically) — pass an explicit `prompt` to
    check alignment against different wording. Advisory only (ADR 0004):
    a low score is a signal to weigh, never a reason this tool itself
    blocks anything.
    """
    try:
        state = session.current()
    except NoActiveSession as exc:
        return {"ok": False, "error": "no_active_session", "message": str(exc)}

    if not any(c["path"] == path for c in state.candidates):
        return {
            "ok": False,
            "error": "unknown_candidate",
            "message": f"{path} is not a candidate from the current session",
        }

    scoring_prompt = prompt if prompt is not None else state.brief
    try:
        clip_score = score_image(PILImage.open(path), scoring_prompt)
    except PipelineExecutionError as exc:
        return {"ok": False, "error": "pipeline_failed", "message": str(exc)}

    return {"ok": True, "path": path, "prompt": scoring_prompt, "clip_score": clip_score}


def register(mcp: "MCPServer") -> None:
    mcp.tool(structured_output=False)(score_image_tool)
