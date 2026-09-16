# ADR 0003: Iteration cap of 5, shared across all fix types

**Status:** Accepted

## Context

The critique loop needs a hard stop enforced in code, not just an agent
instruction — a misjudged "good enough" could otherwise burn GPU time
indefinitely (`docs/PRODUCTION_READINESS.md`'s "iteration cap enforced in
code" item). The cap also has to cover every kind of fix action
(`reprompt` + regenerate, `inpaint`, param adjustment), not just one.

## Decision

Cap total rounds at 5, with every fix action drawing from one shared
counter rather than a separate budget per fix type. Five rounds covers the
common case — diagnose → inpaint → recheck is usually 1–2 rounds — without
letting an ambiguous or genuinely unfixable brief run away. A single shared
counter keeps the cap a one-line server-side check instead of several
independent ones that could each look fine in isolation while the session
as a whole runs long.

## Consequences

If real usage shows 5 is consistently too tight or too loose, revisit using
the iteration-to-convergence distribution from `docs/EVAL.md`'s benchmark
set, not gut feel.
