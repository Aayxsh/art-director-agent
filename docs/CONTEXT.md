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
| **Session** | One full brief-to-final-image run, spanning possibly many iterations | Settled |
| **Cap-out** | When a session hits the iteration cap without the agent confirming any candidate as Acceptable; the session still returns the best-scoring candidate by CLIP score, explicitly flagged as unconfirmed — see ADR 0005 | Settled |

## Open questions for the next `/grill-with-docs` session

- None currently — all three questions from the previous session were
  resolved in this one (see Decisions log).

## Decisions log

*(one line per resolved decision, pointing to the ADR with the full reasoning)*

- Iteration cap is 5, shared across all fix types — [ADR 0003](adr/0003-iteration-cap-of-five.md)
- `score_image` is advisory, agent vision judgment is the primary stopping signal — [ADR 0004](adr/0004-score-image-advisory-not-gate.md)
- Cap-out returns the best-scoring candidate, explicitly flagged — [ADR 0005](adr/0005-cap-hit-fallback.md)
