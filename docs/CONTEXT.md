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
| **Fix** | An action taken in response to a defect: reprompt, inpaint, or param adjustment | TBD — resolve which defect types map to which fix in session |
| **Acceptable** | The threshold at which the loop stops and upscales | TBD — numeric CLIP threshold? human-in-the-loop approval? both? |
| **Iteration cap** | Max rounds before the loop force-stops regardless of score | TBD — cost/latency tradeoff, needs a number and a rationale |
| **Session** | One full brief-to-final-image run, spanning possibly many iterations | Settled |

## Open questions for the next `/grill-with-docs` session

- What happens when the iteration cap is hit and the result still isn't
  acceptable — return the best-scoring candidate, or fail explicitly?
- Does `inpaint` get its own retry budget, or does it count against the
  overall iteration cap?
- Is `score_image` advisory (agent can override it) or a hard gate?

## Decisions log

*(one line per resolved decision, pointing to the ADR with the full reasoning)*

- —
