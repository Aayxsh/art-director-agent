import io
import uuid
from pathlib import Path

from mcp.server.mcpserver import Image as MCPImage
from PIL import Image as PILImage

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
