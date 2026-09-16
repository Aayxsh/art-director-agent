from pathlib import Path

import pytest
from mcp.server.mcpserver import Image as MCPImage
from PIL import Image

import pipeline.upscale as upscale_module
from mcp_server.tools.upscale import upscale_tool


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch, tmp_path):
    def fake_call(*, image, prompt, negative_prompt, guidance_scale, steps, seed):
        return image.copy()

    monkeypatch.setattr(upscale_module, "_default_pipeline_call", fake_call)
    monkeypatch.setattr("mcp_server.tools._media.OUTPUT_DIR", tmp_path)


def _metadata(result: list) -> dict:
    return next(item for item in result if isinstance(item, dict))


def _images(result: list) -> list[MCPImage]:
    return [item for item in result if isinstance(item, MCPImage)]


def _source_image(path, size=(64, 64)) -> str:
    img = Image.new("RGB", size, color="green")
    img.save(path)
    return str(path)


def test_upscale_tool_returns_a_larger_saved_image(tmp_path):
    source = _source_image(tmp_path / "source.png", size=(100, 100))

    result = upscale_tool(source, prompt="a red bicycle", scale=2.0)

    metadata = _metadata(result)
    assert metadata["ok"] is True
    assert metadata["scale"] == 2.0
    assert metadata["source_path"] == source
    saved = Path(metadata["path"])
    assert saved.exists()
    assert Image.open(saved).size == (200, 200)
    assert len(_images(result)) == 1


def test_upscale_tool_does_not_require_an_active_session(tmp_path):
    source = _source_image(tmp_path / "source.png")

    result = upscale_tool(source, prompt="a red bicycle")

    assert _metadata(result)["ok"] is True


def test_upscale_tool_returns_structured_error_for_invalid_input(tmp_path):
    source = _source_image(tmp_path / "source.png")

    result = upscale_tool(source, prompt="a red bicycle", scale=10.0)

    assert result == {
        "ok": False,
        "error": "invalid_input",
        "message": "scale must be between 1.0 and 4.0",
    }


def test_upscale_tool_returns_structured_error_when_pipeline_fails(tmp_path, monkeypatch):
    source = _source_image(tmp_path / "source.png")

    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(upscale_module, "_default_pipeline_call", failing_call)

    result = upscale_tool(source, prompt="a red bicycle")

    assert result["ok"] is False
    assert result["error"] == "pipeline_failed"
