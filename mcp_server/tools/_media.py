import io
import uuid
from pathlib import Path

from mcp.server.mcpserver import Image as MCPImage
from PIL import Image as PILImage

from eval.scoring import PipelineExecutionError, score_image

OUTPUT_DIR = Path("outputs")
THUMBNAIL_MAX_EDGE = 768
THUMBNAIL_QUALITY = 90


def save_image(image: PILImage.Image) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
    image.save(path)
    return path


def thumbnail(image: PILImage.Image) -> MCPImage:
    thumb = image.copy()
    thumb.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
    return MCPImage(data=buf.getvalue(), format="jpeg")


def score_or_none(image: PILImage.Image, prompt: str) -> float | None:
    """CLIP score against `prompt`, or None if scoring fails.

    Advisory (ADR 0004) — a scoring failure must never block the primary
    generation/inpaint result from succeeding.
    """
    try:
        return score_image(image, prompt)
    except PipelineExecutionError:
        return None
