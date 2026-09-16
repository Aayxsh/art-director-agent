# ADR 0007: Implicit global session, no session_id

**Status:** Accepted

## Context

ADR 0003's iteration cap (5 rounds, shared across fix types) has to be
enforced server-side (`docs/PRODUCTION_READINESS.md`), which means some
notion of "the current session" has to persist across otherwise-stateless
MCP tool calls. Two shapes were available: an explicit `session_id` the
agent creates and threads through every call, or an implicit
single-active-session tracked as server-side global state.

## Decision

Implicit global state (`mcp_server/session.py`), no `session_id`.
`generate_image_tool` takes a `continue_session: bool` flag instead:
`False` (the default) starts a fresh session at that prompt; `True`
consumes the next round of whatever session is already active. ADR 0002
already scopes this project to a single local GPU running one session at
a time — explicit session-id plumbing only pays for itself under
concurrent/multi-tenant use, which is out of scope for v1.

## Consequences

If the project ever needs concurrent sessions (ADR 0002's own
consequences section already flags this as a "different ADR" trigger),
`continue_session: bool` doesn't extend to that — it becomes an explicit
`session_id: str`, a breaking change to every tool in `mcp_server/tools/`
that touches session state. Revisit only if that requirement actually
shows up.
