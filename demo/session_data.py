import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExampleSession", "best_candidate", "list_examples", "load_example"]


@dataclass(frozen=True)
class ExampleSession:
    slug: str
    brief: str
    rounds: list[list[dict]]
    upscaled_path: str | None


def list_examples(examples_dir: Path) -> list[str]:
    """Slugs of available curated example sessions, sorted."""
    return sorted(p.stem for p in examples_dir.glob("*.json"))


def load_example(examples_dir: Path, slug: str) -> ExampleSession:
    data = json.loads((examples_dir / f"{slug}.json").read_text())
    return ExampleSession(
        slug=slug,
        brief=data["brief"],
        rounds=data["rounds"],
        upscaled_path=data.get("upscaled_path"),
    )


def best_candidate(session: ExampleSession) -> dict | None:
    """The highest CLIP-scoring candidate across the whole session, or None."""
    scored = [c for round_ in session.rounds for c in round_ if c.get("clip_score") is not None]
    return max(scored, key=lambda c: c["clip_score"]) if scored else None
