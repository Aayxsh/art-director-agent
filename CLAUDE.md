# CLAUDE.md

## 1. Project Overview

**art-director-agent** — a closed-loop image generation agent.

An MCP server wraps a diffusers-based SDXL pipeline (base + refiner, inpainting,
upscaling) and exposes it as tools. Claude acts as the orchestrator: generate →
look at the result (vision) → diagnose what's wrong against the brief → call the
right fix tool → repeat until a quality bar is hit. The point is not "an agent
that calls diffusers" — it's an agent that knows when its own output is wrong
and can fix it. That eval-driven loop is the whole thesis of the project.

**Why it exists:** portfolio piece for AI/ML job applications. Demonstrates:
- MCP server design (tool schemas, not just tool calling)
- Multi-step agentic control loop with a real stopping condition
- Applied diffusion model work (SDXL, inpainting, ControlNet if time allows)
- Evaluation-driven engineering (the same instinct as the agentic-rag-eval project)

**Core components:**
- `mcp_server/` — the MCP server exposing `generate_image`, `inpaint`, `upscale`,
  `score_image`, `get_history`
- `pipeline/` — diffusers wrapper (model loading, inference, image I/O)
- `eval/` — scoring (CLIP similarity / aesthetic score) + iteration logging
- `demo/` — deployed Streamlit/Gradio app showing the before/after loop
- `docs/` — CONTEXT.md (domain language, maintained by `/grill-with-docs`) and
  `docs/adr/` (architecture decisions)

**Out of scope for v1:** training/fine-tuning models, multi-user auth, anything
beyond a single-session critique loop. Note it in `docs/adr/` if that changes —
don't silently scope-creep.

---

## 2. Project Status

Update this section at the end of every session. One line per phase, status only.

| Phase | Description | Status |
|---|---|---|
| 0 | Grilled the design — CONTEXT.md + ADRs exist | Done |
| 1 | MCP server scaffold + `generate_image` tool (base SDXL, no fixes yet) | Done |
| 2 | Critique loop: agent inspects output, diagnoses problems | Done |
| 3 | `inpaint` + `upscale` tools wired into the loop | Done |
| 4 | `score_image` + `get_history`, iteration logging for the demo | Done |
| 5 | Deployed demo + README with before/after grid | Done (demo built + deploy-ready; not actually pushed public — see below) |

**Current focus:** All 5 core phases done. Remaining optional work is listed in `README.md`'s Roadmap (full `EVAL.md` benchmark at scale, dedicated inpainting checkpoint, adversarial critic, ControlNet, batch benchmark mode).
**Known issues / open questions:** Running base + inpaint + refiner pipelines back-to-back in one process can exhaust system RAM (31GB total) and get OS-killed, even though GPU VRAM is fine — observed during Phase 3's GPU test validation. Run GPU-marked tests for different pipeline modules in separate `pytest` invocations, not one combined `-m gpu` sweep, until this is understood better. `demo/app.py` is built, tested (headless `AppTest`), and deploy-ready for Streamlit Community Cloud — but this repo has never been pushed to a public GitHub remote and no hosting account was connected, deliberately (ADR 0012): that's the user's action to trigger, not something done automatically mid-phase.

---

## 3. Coding Style

- **Python**, type-hinted throughout — no bare `dict`/`Any` where a `TypedDict`
  or dataclass is cheap to write instead.
- Format with `ruff format`; lint with `ruff check`. Run both before committing.
- Docstrings on every public function: one line of intent, not a restatement
  of the signature.
- MCP tool functions: validate inputs explicitly, return structured errors
  (not raw exceptions) — a tool that fails silently or returns a stack trace
  to the agent is a debugging nightmare later. Every tool needs a docstring
  the *model* can use to decide when to call it, written for that audience.
- Tests live next to the code they test (`pipeline/generate.py` →
  `pipeline/test_generate.py`), driven by `/tdd` — red, green, refactor, one
  vertical slice at a time. Don't write implementation ahead of a failing test.
- MCP-tool-layer tests fake the underlying pipeline call by monkeypatching
  that pipeline module's `_default_pipeline_call` (`pipeline.generate`,
  `pipeline.inpaint`, `pipeline.upscale`, ...) — the sanctioned internal
  seam, not a per-tool convention to reinvent. Shared timeout/seed/error
  mechanics live in `pipeline/_runtime.py`; shared thumbnail/disk-save
  mechanics live in `mcp_server/tools/_media.py` — extend those, don't
  duplicate them into a new pipeline operation or tool.
- Commit after each completed phase (matching the agentic-rag-eval workflow) —
  small, reviewable, message states what changed and why.
- No giant files. If something creeps past ~300 lines, that's a
  `/improve-codebase-architecture` trigger, not a "later" problem.

---

## 4. Workflow — when to invoke which skill

1. **Before writing code for a new phase:** run `/grill-with-docs`. Let it
   interrogate the plan, update `docs/CONTEXT.md` with the domain language
   (what counts as a "fix," what "good enough" means for the scoring
   threshold, tool naming), and write an ADR for any real decision (e.g. why
   CLIP score over a learned aesthetic model, why iteration cap is 5).
2. **Building the phase:** `/implement` driving `/tdd` — test-first, one
   vertical slice at a time. Don't let it batch three tools into one
   uncommitted blob.
3. **Phase feels done:** `/improve-codebase-architecture` — checks structure
   against `docs/CONTEXT.md` and the ADRs, catches entropy before it compounds.
4. **Before committing / before it goes in the portfolio:** `/thermo-nuclear-code-quality-review`
   for the ambitious, harsh pass — giant files, spaghetti, missed
   simplifications. Treat its findings as judgment calls against this file's
   stated conventions, not hard blockers.
5. **Stuck or missing a workflow piece:** `/find-skills`.

Maintenance rule: any time a skill is added, removed, or this workflow order
changes, update this section in the same commit.

---

## 5. Project Structure

```
art-director-agent/
├── CLAUDE.md
├── docs/
│   ├── CONTEXT.md          # domain language, maintained by /grill-with-docs
│   └── adr/                # one file per real architectural decision
├── mcp_server/
│   ├── server.py
│   ├── session.py           # server-side session/round-cap tracking (ADR 0003, ADR 0005)
│   └── tools/               # generate_image, inpaint, upscale, score_image, get_history
│       └── _media.py         # shared thumbnail + disk-save helpers
├── pipeline/
│   ├── _runtime.py           # shared timeout/seed/error-wrapping (used by every pipeline op)
│   ├── generate.py
│   ├── inpaint.py            # bbox mask, shares base pipeline components (ADR 0008)
│   ├── upscale.py            # refiner pass + resize, no super-res model (ADR 0009)
│   └── test_*.py            # TDD, colocated
├── eval/
│   ├── scoring.py           # CLIP / aesthetic score
│   └── session_log.py       # per-iteration history for the demo (was logging.py — renamed, shadowed stdlib)
├── demo/
│   ├── app.py                 # Streamlit — thin rendering layer, untested (composition root)
│   ├── session_data.py        # pure data logic (list/load/score sessions) — the tested seam
│   ├── curate_examples.py     # utility: convert real eval/results/ sessions into demo/examples/
│   └── examples/               # committed, curated example sessions (ADR 0012) — not gitignored,
│                                # unlike eval/results/ and outputs/
└── README.md                 # the portfolio-facing writeup — metrics, not adjectives
```
