# ADR 0009: `upscale` is a refiner pass plus a plain resize, not a super-resolution model

**Status:** Accepted

## Context

`README.md`'s pipeline names `upscale` as the final step after a
candidate is Acceptable. A dedicated pixel-upscaling model (e.g.
`stabilityai/stable-diffusion-x4-upscaler`) was the obvious-sounding
choice, but wasn't in ADR 0002's VRAM budget. Checked before deciding:
weights alone are modest (~2GB fp16), but the model conditions on the
low-res image at *output* resolution, so working memory scales with
output size rather than staying in latent space — real-world reports
show OOM upscaling past 256×256 input even on a 10GB card, and 512×512
input needing 24GB with `xformers`. Stacked on top of the already-resident
base/refiner/inpaint pipelines, a 1024×1024→4096×4096 upscale on this
project's 16GB card is a likely OOM, not a safe bet.

## Options considered

1. **Dedicated super-resolution model** (`stable-diffusion-x4-upscaler` or
   similar) — real pixel-level upscaling quality, high OOM risk at this
   project's target resolution given the stacked VRAM budget.
2. **A lightweight GAN upscaler** (e.g. Real-ESRGAN, ~2-4GB VRAM,
   tileable so VRAM stays roughly constant regardless of input size) —
   fits comfortably, but is a new model dependency ADR 0002 never named.
3. **SDXL refiner pass (img2img, same resolution, adds detail) + a plain
   Lanczos resize** (PIL, CPU, no VRAM) to actually change pixel
   dimensions — the refiner was already in ADR 0002's budget as part of
   "base + refiner"; nothing new to load.

## Decision

Option 3. It's the only one that adds zero new VRAM commitment and zero
new dependencies, and "refiner + resize" is arguably closer to what
`README.md`'s "upscale the winner" line was describing all along than a
literal super-resolution model would be — the refiner *does* the quality
work in the pipeline this project already describes; the resize is just
the pixel-dimension increase that phrase also implies.

## Consequences

Detail added by the refiner is at native resolution, then upsampled by
Lanczos — not genuinely higher-resolution detail the way a real
super-resolution model would produce. If output quality proves
insufficient, option 2 (Real-ESRGAN) is the next thing to try: it fits
the existing VRAM budget and only replaces
`pipeline/upscale.py`'s `_real_pipeline_call`, not `upscale()`'s public
interface (`source_path`, `scale`, `prompt` stay the same).
