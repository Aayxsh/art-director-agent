# ADR 0012: Demo is a replay-only viewer over a curated, committed example set

**Status:** Accepted

## Context

`README.md` already described the demo as replaying "the iteration history"
(ADR 0011), but Phase 5 needed to decide what "deployed" means given ADR
0002's scope: a public host has no GPU, and ADR 0002 never budgeted for
hosted inference. `eval/results/*.json` and `outputs/*.png` are correctly
gitignored (generated artifacts), so a deployed demo needs something else to
show.

## Decision

`demo/app.py` never triggers generation — it only reads session JSON from
`demo/examples/`, a small, committed, hand-curated set (3 real sessions as of
this phase) distinct from the gitignored raw logs. Each curated file extends
the raw `eval/session_log.py` schema with one additional field, `upscaled_path`
(null when the session's result was never accepted/upscaled) — `upscale`
itself stays decoupled from session tracking (ADR 0009), so this field is
demo-specific curation, not a change to the core logging schema. Images are
converted to JPEG for the committed copies (full-res PNGs from `outputs/`
are ~1.5-2.8MB each; the curated set converts to ~2MB total).

Because the demo needs no GPU at request time, it's deployable on a free
host (Streamlit Community Cloud, linked directly to the GitHub repo) — but
creating that public repo and connecting a hosting account are the user's
own actions to trigger, not something done automatically as part of a build
phase.

## Consequences

Refreshing the demo's examples is a manual curation step (`demo/curate_examples.py`),
not automatic — new real sessions don't appear in the deployed demo until
someone re-runs curation and commits the result. Fine for a portfolio
project's small, deliberately-chosen example set; would need revisiting if
this ever needed to show live or frequently-refreshed sessions.
