# ADR 0011: Session history persists to disk, not just in-process state

**Status:** Accepted

## Context

`mcp_server/session.py` is in-process global state (ADR 0007) — gone on
server restart, and unreachable from another process. `get_history` (this
phase) is called by the agent within the same server process, so it can
read that live state directly. But the Phase 5 demo (`streamlit run
demo/app.py`) runs as its own separate process from the MCP server the
agent talks to, and can't read Python globals across that boundary no
matter what. `CLAUDE.md`'s project structure already named `eval/session_log.py`
(originally `eval/logging.py` — renamed, see Consequences)
"per-iteration history for the demo" as a file distinct from
`mcp_server/session.py`, and `.gitignore` already anticipated
`eval/results/*.json`.

## Decision

`mcp_server/session.py`'s `start_new()` and `record_round()` call into
`eval/session_log.py` internally — one JSON file per session, written as each
round completes, so persistence can't be forgotten by a caller. `get_history`
itself still reads live session state (no reason to round-trip through
disk for a same-process read); the persisted files exist purely for the
Phase 5 demo to read later, potentially across many past sessions, not
just the live one.

## Consequences

Every session run leaves a JSON file in `eval/results/` (gitignored).
Phase 5's demo needs to pick a session (e.g. most recent, or a picker
across files) rather than assuming exactly one — a small scope addition
to what was originally described as replaying "the iteration history"
singular.

**Post-Phase-5 rename:** `eval/logging.py` was renamed to
`eval/session_log.py`. Its name shadowed Python's stdlib `logging`
module whenever a script ran with `eval/` itself on `sys.path` (e.g.
`python eval/some_script.py`, run directly rather than via pytest or
`python -m`) — breaking anything downstream that needs real `logging`
(PIL, diffusers, transformers all use it internally). Caught while
running the `docs/EVAL.md` benchmark, whose runner script lives in
`eval/`. The persistence decision above is unchanged, just the filename.
