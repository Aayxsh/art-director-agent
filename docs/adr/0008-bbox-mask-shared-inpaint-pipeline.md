# ADR 0008: Normalized bounding-box mask, shared-component inpaint pipeline

**Status:** Accepted

## Context

`inpaint` needs to know which region of a candidate to fix (a defect the
agent identified via vision, per ADR 0004), and diffusers' inpainting
pipeline needs a mask image (white = inpaint, black = keep). It also
needs an actual inpainting-capable pipeline — SDXL's proper inpainting
quality normally comes from a separately-finetuned checkpoint (a 9-channel
UNet, distinct weights from the base pipeline's 4-channel UNet), which
would not share VRAM with the already-loaded base pipeline the way ADR
0002 assumed ("the inpainting pipeline shares most of those weights
rather than doubling the footprint").

## Options considered

**Mask input:**
1. A real segmentation model (CLIPSeg/SAM) driven by a text description of
   the defect — more precise, but a new model dependency ADR 0002's VRAM
   budget never accounted for.
2. A normalized bounding box (`x0, y0, x1, y1` in 0-1 coordinates) that the
   tool turns into a rectangle mask — resolution-independent, no new
   model, matches what a vision-critiquing LLM can actually specify
   (roughly *where*, not a pixel-perfect boundary).

**Inpainting pipeline:**
1. The dedicated inpainting-finetuned checkpoint — better fix quality,
   but a separate ~5GB UNet that breaks ADR 0002's shared-weights
   assumption and needs its own download.
2. `StableDiffusionXLInpaintPipeline.from_pipe(base_pipe)`, reusing the
   already-loaded base pipeline's components — confirmed as a real
   diffusers pattern (no reload, no duplicate VRAM; only a lightweight
   wrapper object is created). Lower fix quality on some defects, since
   the base UNet wasn't finetuned for inpainting.

## Decision

Bounding box + `from_pipe(base_pipe)`. Both keep Phase 3 inside the VRAM
budget ADR 0002 actually committed to, at zero new model downloads and
zero new dependencies — the option ADR 0002's own risk tolerance implies,
given it was written for a portfolio project on a single consumer GPU,
not a produce quality benchmark.

## Consequences

Fix quality on `inpaint` may be visibly weaker than a dedicated
inpainting checkpoint would give, especially on defects requiring strong
structural change (e.g. malformed hands — exactly ADR 0004's motivating
example). If `docs/EVAL.md`'s human spot-check shows this is a real
problem, the fix is swapping `_load_real_inpaint_pipeline` in
`pipeline/inpaint.py` for the dedicated checkpoint — the rest of the
module (bbox validation, mask building, seam/timeout handling) doesn't
change.
