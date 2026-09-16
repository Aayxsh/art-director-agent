# ADR 0002: Local GPU inference, not hosted

**Status:** Accepted

## Context

The critique loop needs SDXL base + refiner, an inpainting pass, an upscaler,
and a CLIP model for `score_image` — potentially all resident at once during
a session. This has to run somewhere, and where affects latency, cost, and
how much of the "production readiness" story (rate limiting, timeouts,
spending caps) is even relevant.

Available hardware: RTX 5070 Ti (16GB GDDR7 VRAM), 32GB system RAM.

## Options considered

1. **Hosted inference (Replicate, Modal, RunPod)** — no local VRAM ceiling,
   easier to scale if the demo goes public, but adds real per-call cost,
   network latency on every tool call, and an external dependency the whole
   critique loop now blocks on.
2. **Local GPU** — SDXL base + refiner in fp16 runs at ~7-8GB VRAM; the
   inpainting pipeline shares most of those weights rather than doubling the
   footprint; a CLIP scorer adds ~1-2GB. All of it fits comfortably inside
   16GB without attention slicing or CPU offloading, at standard 1024×1024
   resolution, with system RAM well above what's needed for the MCP server
   and demo app running alongside.

## Decision

Local GPU inference on the RTX 5070 Ti. The VRAM budget covers the full
pipeline (base + refiner + inpaint + CLIP scorer) simultaneously without
needing memory-saving tradeoffs that would add latency, and it removes
per-call cost and network latency from every iteration of the critique loop.

## Consequences

- No hosted-inference spending cap needed in `docs/PRODUCTION_READINESS.md` —
  drop that item, it doesn't apply locally
- Generate the 2-3 candidates per round sequentially rather than batched,
  since multiple pipelines stay loaded at once; revisit only if profiling
  shows this is an actual bottleneck, not preemptively
- Confirm the installed PyTorch build has stable Blackwell (sm_120) support
  before relying on this — a nightly-only dependency would be a fragile
  foundation for a portfolio project meant to just work when someone clones it
- If the demo ever needs to go public/multi-user rather than a local run
  captured on video, this decision gets revisited — a single local GPU
  can't serve concurrent sessions, and that's a different ADR
