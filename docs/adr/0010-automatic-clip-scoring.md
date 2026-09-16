# ADR 0010: Every candidate is scored automatically, not only on request

**Status:** Accepted

## Context

ADR 0005 promised cap-hit ranks candidates by CLIP score once `score_image`
exists. But ADR 0001 made `score_image` its own separate, agent-called MCP
tool, and ADR 0004 made it advisory — the agent isn't required to call it.
If scoring stayed purely opt-in, some candidates could reach cap-out with
no score recorded, leaving gaps in exactly the ranking ADR 0005 needs.

## Options considered

1. **Opt-in only** — `score_image` stays the sole place a score gets
   computed; cap-hit ranks only among whatever happens to be scored,
   falling back to "most recent" otherwise. Simpler, but reintroduces the
   gap ADR 0005 was written to close.
2. **Automatic** — `generate_image` and `inpaint` compute a CLIP score for
   every candidate as part of their own response. `score_image` still
   exists separately (ADR 0001) for the agent to re-check a specific
   candidate or test different wording, but isn't the only source.

## Decision

Automatic (option 2), scored against the session's brief (not each
candidate's own narrower `prompt` — inpaint's `prompt` describes only the
masked region, and a "reprompt" fix round may use revised wording; the
brief is the one stable target that means "what the user actually asked
for" across every round). Uses `openai/clip-vit-large-patch14` in fp16 —
measured at ~1.0-1.5GB VRAM, comfortably inside the ~6-7GB headroom left
after ADR 0002's base/refiner/inpaint stack (checked before choosing,
given ADR 0009 already got burned once assuming an under-researched
model's VRAM would be fine).

A scoring failure is non-fatal: `generate_image`/`inpaint` still succeed
with `clip_score: null` for that candidate (ADR 0004 — advisory, never a
gate). `mcp_server/session.py`'s `best_scoring_round()` falls back to the
most recent round if nothing in the session has a score at all.

## Consequences

Every `generate_image`/`inpaint` call now also runs a CLIP forward pass —
extra latency per round (small relative to SDXL inference itself), and
the CLIP model stays resident in VRAM for the rest of the process once
loaded. `docs/adr/0005-cap-hit-fallback.md` is updated with the resulting
"best-scoring round, not literally one candidate" ranking shape.
