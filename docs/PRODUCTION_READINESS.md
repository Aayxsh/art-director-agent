# Production readiness

Most portfolio image-gen demos skip this entirely. Doing even a scoped-down
version is one of the more effective, low-cost ways to signal you think past
"it works on my machine" — and it's a natural fit given the loop itself can
run away in cost/latency if left unchecked.

Scoped to what this project actually needs (skip anything that's SaaS-only
and doesn't apply to a single-user demo — no payments or subscriptions here).
Per ADR 0002 (local GPU inference), hosted-inference spending caps don't
apply — removed from this list rather than left as a stale checkbox:

- [ ] **Iteration cap enforced in code**, not just as an agent instruction —
      the agent can misjudge "good enough" and loop past a sane budget if
      nothing stops it server-side
- [ ] **Rate limit `generate_image` calls** — GPU inference is the expensive
      part; a runaway loop or a public demo without limits gets costly fast
- [ ] **Timeout handling on pipeline calls** — SDXL inference can hang or OOM;
      the MCP tool should return a structured error to the agent, not crash
      the server or leave the agent waiting indefinitely
- [ ] **Cache identical (prompt, seed, params) calls** — the agent may retry
      a configuration it's already tried; don't regenerate for free
- [ ] **Structured logging per iteration** — already planned in `eval/session_log.py`;
      make sure failures log too, not just successes
- [ ] **Input limits** — cap prompt length and, for `inpaint`, uploaded mask/
      image size, before they hit the pipeline
- [ ] **Concurrent session test** if the demo is public-facing — Streamlit/
      Gradio's default queuing behavior under 2–3 simultaneous users is worth
      checking once before calling the demo "deployed"

Each checked box is a sentence in the README's "What I learned" section, not
just a checkbox — "added a server-side iteration cap after the agent looped
9 times on an ambiguous brief" is a better portfolio line than the checkbox
itself.
