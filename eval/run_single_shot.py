import json
import time
from pathlib import Path

from mcp_server.tools.generate_image import generate_image_tool

prompts = json.loads(Path("eval/benchmark_prompts.json").read_text())
results = []

for p in prompts:
    start = time.time()
    result = generate_image_tool(p["prompt"], num_images=1, seed=p["id"])
    latency = time.time() - start
    meta = next(x for x in result if isinstance(x, dict))
    candidate = meta["candidates"][0]
    results.append(
        {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "path": candidate["path"],
            "clip_score": candidate["clip_score"],
            "latency_s": round(latency, 2),
        }
    )
    score = candidate["clip_score"]
    print(f"[{p['id']:2d}/{len(prompts)}] {p['category']:12s} clip={score:.1f} lat={latency:.1f}s")

Path("eval/results/single_shot_benchmark.json").write_text(json.dumps(results, indent=2))
print("done, wrote eval/results/single_shot_benchmark.json")
