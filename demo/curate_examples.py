import json
from pathlib import Path

from PIL import Image

SESSIONS = [
    {
        "log": "eval/results/20260916T062047675238-3bbfef85.json",
        "slug": "01-bouquet",
        "upscaled_path": "outputs/3e602c03508e4ae8ac137d2b6aa13616.png",
    },
    {
        "log": "eval/results/20260916T062214481947-83bbf7cb.json",
        "slug": "02-handshake",
        "upscaled_path": None,
    },
    {
        "log": "eval/results/20260916T062707795315-7678af34.json",
        "slug": "03-neon-sign",
        "upscaled_path": None,
    },
]

IMAGES_DIR = Path("demo/examples/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def convert(src: str, slug: str, tag: str) -> str:
    src_path = Path(src)
    dest_name = f"{slug}-{tag}.jpg"
    dest_path = IMAGES_DIR / dest_name
    img = Image.open(src_path).convert("RGB")
    img.save(dest_path, format="JPEG", quality=88)
    return f"images/{dest_name}"


for session in SESSIONS:
    data = json.loads(Path(session["log"]).read_text())
    out_rounds = []
    path_map: dict[str, str] = {}
    for round_idx, round_candidates in enumerate(data["rounds"]):
        out_candidates = []
        for cand_idx, candidate in enumerate(round_candidates):
            new_candidate = dict(candidate)
            new_path = convert(
                candidate["path"], session["slug"], f"r{round_idx + 1}c{cand_idx + 1}"
            )
            path_map[candidate["path"]] = new_path
            new_candidate["path"] = new_path
            if "source_path" in new_candidate:
                new_candidate["source_path"] = path_map.get(
                    new_candidate["source_path"], new_candidate["source_path"]
                )
            out_candidates.append(new_candidate)
        out_rounds.append(out_candidates)

    upscaled_path = None
    if session["upscaled_path"]:
        upscaled_path = convert(session["upscaled_path"], session["slug"], "upscaled")

    out_data = {
        "brief": data["brief"],
        "rounds": out_rounds,
        "upscaled_path": upscaled_path,
    }
    out_file = Path(f"demo/examples/{session['slug']}.json")
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"wrote {out_file} ({len(out_rounds)} rounds)")

total = sum(f.stat().st_size for f in IMAGES_DIR.glob("*.jpg"))
print(f"total image size: {total / 1e6:.1f} MB")
