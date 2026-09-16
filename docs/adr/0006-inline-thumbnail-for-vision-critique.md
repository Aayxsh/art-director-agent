# ADR 0006: Inline JPEG thumbnails for vision critique, full-res PNG stays on disk

**Status:** Accepted

## Context

ADR 0004 makes the agent's own vision critique the primary judge of a
candidate — but `generate_image_tool` originally returned only a
filesystem path, which an MCP client can't view. The tool needs to return
actual image content. Measured on the real SDXL pipeline (RTX 5070 Ti): a
full 1024×1024 PNG is ~1.7MB (~2.2MB base64), so 2-3 candidates per round
is 4.5-6.7MB of tool-response payload — expensive in both transport and
vision tokens, every round, for a loop capped at 5 rounds per session.

## Decision

Return a resized JPEG (768px on the long edge, quality 90) as inline
`Image` content for each candidate, alongside a trailing metadata dict.
The full-resolution PNG is unchanged — still saved to `outputs/` and
referenced by `path` in the metadata, for `upscale` (Phase 3) and the
demo's history replay (Phase 5). 768px/q90 was chosen as a middle
ground: small enough to be cheap every round, large enough to keep
finger/hand-level detail legible for the exact defect class ADR 0004's
rationale is built around.

## Consequences

If vision critique turns out to miss defects at this resolution, the
fix is a two-constant change (`THUMBNAIL_MAX_EDGE`, `THUMBNAIL_QUALITY`
in `mcp_server/tools/generate_image.py`), not a redesign.
