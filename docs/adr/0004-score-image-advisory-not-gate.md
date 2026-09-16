# ADR 0004: `score_image` is advisory, not a hard gate

**Status:** Accepted

## Context

`score_image` (CLIP prompt–image alignment) runs every round and could be
used as a hard numeric threshold to decide when the loop stops. But CLIP
alignment doesn't catch the exact defects this project exists to catch —
`docs/EVAL.md` notes a six-fingered hand can score fine on prompt alignment
despite being visibly wrong. Gating purely on CLIP score would let that
failure mode pass silently, undermining the vision-critique loop that's the
project's core thesis (`README.md`, "Why this exists").

## Decision

The agent's own vision-based critique of each candidate is the primary
judge of "good enough." `score_image` is a secondary, advisory signal the
agent can weigh but is not required to obey — there is no hard numeric CLIP
threshold gating the loop, and no separate hard CLIP floor as a safety net
either. The agent's vision judgment is trusted on its own.

## Considered options

- **Hard numeric CLIP threshold** — rejected per the six-fingers case above.
- **Human-in-the-loop approval** — rejected; out of v1 scope per `CLAUDE.md`
  (single-session, no human approval step). The human spot-check in
  `docs/EVAL.md` is offline benchmarking, not a runtime gate.

## Consequences

`get_history` / `eval/session_log.py` must still record CLIP score every round
even though it isn't gating — it's the only scalar comparable across
iterations, and it's what the cap-hit fallback (ADR 0005) ranks by.
