# ADR 0005: Cap-hit fallback returns the best-scoring candidate, flagged

**Status:** Accepted

## Context

Because `score_image` is advisory (ADR 0004), a session can hit the
iteration cap (ADR 0003) without the agent ever judging a candidate
Acceptable. The session still has to return something.

## Decision

Return the candidate with the highest CLIP score across all rounds,
explicitly flagged that the cap was hit without a confirmed pass. Never
return it silently as a success, and never discard the work as a hard
failure — real GPU time already produced real candidates, and the
best-scoring one is a reasonable result even if not a confirmed one.

## Consequences

`get_history` / `eval/logging.py` must retain per-round CLIP score and
agent notes for every candidate generated in a session, not just the most
recent ones, so this pick is possible at cap-out.
