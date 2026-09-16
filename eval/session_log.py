import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("eval/results")

__all__ = ["log_round", "start_session_log"]


def start_session_log(brief: str) -> str:
    """Create a new session log file for the demo (Phase 5) to replay later.

    Written eagerly so the file exists from round 1, independent of
    mcp_server/session.py's in-process state (ADR 0007) — the demo runs
    as a separate process and can't read that state directly.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = RESULTS_DIR / f"{timestamp}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps({"brief": brief, "rounds": []}, indent=2))
    return str(path)


def log_round(log_path: str, candidates: list[dict]) -> None:
    """Append one completed round's candidates to the session log at `log_path`."""
    path = Path(log_path)
    data = json.loads(path.read_text())
    data["rounds"].append(candidates)
    path.write_text(json.dumps(data, indent=2))
