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

**Phase 2 interim note:** `score_image` doesn't exist yet, so there's no
CLIP score to rank by. Until Phase 4, cap-out returns the *most recent*
round's candidates instead of the *best-scoring* ones — same shape
(`cap_hit: true`, never silent success, never discarded work), weaker
ranking. Phase 4 swaps "most recent" for "best-scoring by CLIP" as a
non-breaking change to `mcp_server/tools/generate_image.py`'s
`_cap_hit_result`.
