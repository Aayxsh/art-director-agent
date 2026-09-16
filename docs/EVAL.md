# Evaluation methodology

The core claim of this project is that the critique loop *improves* generation
over single-shot prompting. That claim needs a number, not an adjective — this
doc is how to get one.

## The anchor comparison

Run the same fixed prompt set through two conditions:

1. **Single-shot** — one `generate_image` call, no critique, no fixes. This is
   the baseline every "prompt box" project effectively ships.
2. **Full loop** — the whole critique → fix → re-score cycle, up to the
   iteration cap.

The gap between these two is the entire point of the project. Report it in
the README results table, not just in this doc.

## Benchmark set

A fixed set of ~20–30 prompts, spanning categories known to be hard for SDXL
so the loop has something to actually fix:

- Portraits / hands (the classic failure mode)
- Multi-object scenes (composition, occlusion)
- Text-in-image requests (near-guaranteed single-shot failures)
- Specific style/mood briefs (subjective — needs the human check below)

Keep this set fixed across runs so results are comparable over time, and
version it — if you add prompts later, note in the results which version
produced which numbers.

## Metrics

- **CLIP score** (prompt–image alignment) — cheap, automatic, run every time
- **Iterations to convergence** — how many rounds the loop needed per prompt;
  report the average and the distribution, not just the mean (a few prompts
  that hit the iteration cap and never converge is a more useful signal than
  a clean average hides)
- **Wall-clock latency** — per session, since this is a real cost tradeoff
  a hiring manager will ask about
- **Human spot-check pass rate** — CLIP score doesn't catch everything (six
  fingers can score fine on prompt alignment). Spot-check a sample by eye and
  report a rough pass rate. Be honest about sample size — "8/10 on a manual
  check of 10 prompts" is a fine thing to publish, "94% accurate" from the
  same 10 prompts is not.

## Failure analysis

For any prompt that hits the iteration cap without converging, log why:
was it a bad fix decision, a scoring blind spot, or a genuinely hard prompt?
This section of the eventual writeup — not the metrics table — is usually
what a good interviewer actually asks about.
