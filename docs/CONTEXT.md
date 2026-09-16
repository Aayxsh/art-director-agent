# CONTEXT.md — domain language

Maintained by `/grill-with-docs`. This is the shared vocabulary between you
and the agent — when a term here is ambiguous, that's exactly what a grilling
session should resolve before the next phase starts. Don't let this file go
stale; an outdated glossary is worse than none.

## Core terms

| Term | Definition | Status |
|---|---|---|
| **Brief** | The user's original request for an image, before prompt expansion | Settled |
| **Candidate** | One generated image produced by `generate_image` in a given round | Settled |
| **Defect** | A specific, describable problem with a candidate (not "doesn't look right" — "left hand has six fingers") | Settled |
| **Fix** | An action taken in response to a defect, chosen by defect type: `inpaint` for a localized/region defect (e.g. malformed hands), reprompt + regenerate for a whole-image/conceptual defect (e.g. wrong composition), or guidance/steps adjustment + retry for a global quality defect (e.g. blurriness) | Settled |
| **Acceptable** | The point at which the agent's own vision critique judges a candidate good enough to upscale. `score_image`'s CLIP score is an advisory secondary signal, not a hard gate — see ADR 0004 | Settled |
| **Iteration cap** | Max rounds before the loop force-stops regardless of score: 5, shared across all fix types (no separate budget for `inpaint`) — see ADR 0003 | Settled |
| **Session** | One full brief-to-final-image run, spanning possibly many iterations. Tracked server-side as implicit global state, no `session_id` — `generate_image`'s `continue_session` flag starts fresh (`False`) or advances the active one (`True`) — see ADR 0007 | Settled |
| **Cap-out** | When a session hits the iteration cap without the agent confirming any candidate as Acceptable. Returns the best-scoring candidate by CLIP score, explicitly flagged as unconfirmed, once `score_image` exists (ADR 0005) — until Phase 4, returns the *most recent* round's candidates instead, same flagging | Settled |

## Open questions for the next `/grill-with-docs` session

- None currently.

## Decisions log

*(one line per resolved decision, pointing to the ADR with the full reasoning)*

- Iteration cap is 5, shared across all fix types — [ADR 0003](adr/0003-iteration-cap-of-five.md)
- `score_image` is advisory, agent vision judgment is the primary stopping signal — [ADR 0004](adr/0004-score-image-advisory-not-gate.md)
- Cap-out returns the best-scoring candidate, explicitly flagged — [ADR 0005](adr/0005-cap-hit-fallback.md)
- Inline JPEG thumbnails (768px, q90) for vision critique; full-res PNG stays on disk — [ADR 0006](adr/0006-inline-thumbnail-for-vision-critique.md)
- Session tracked as implicit global state, no `session_id` — [ADR 0007](adr/0007-implicit-global-session.md)
- `inpaint` takes a normalized bounding box, reuses the base pipeline's shared components (no dedicated inpainting checkpoint) — [ADR 0008](adr/0008-bbox-mask-shared-inpaint-pipeline.md)
- `upscale` is a refiner pass + plain resize, not a super-resolution model; doesn't count against the round cap — [ADR 0009](adr/0009-upscale-is-refiner-plus-resize.md)
