from typing import TYPE_CHECKING

from mcp_server.session import NoActiveSession, session

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def get_history_tool() -> dict:
    """Return the current session's full round-by-round history.

    Every round so far, in order, with each candidate's saved metadata
    (`path`, `seed`, `prompt`, `clip_score`, ...) — no images re-attached,
    the agent already saw thumbnails when each round was generated.
    Fails with `no_active_session` if nothing was started via
    generate_image first.
    """
    try:
        state = session.current()
    except NoActiveSession as exc:
        return {"ok": False, "error": "no_active_session", "message": str(exc)}

    return {
        "ok": True,
        "brief": state.brief,
        "rounds_used": state.rounds_used,
        "rounds": session.all_rounds(),
    }


def register(mcp: "MCPServer") -> None:
    mcp.tool(structured_output=False)(get_history_tool)
