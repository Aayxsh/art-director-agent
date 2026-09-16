from pathlib import Path

import pytest
from mcp.server.mcpserver import Image as MCPImage
from PIL import Image

import pipeline.inpaint as inpaint_module
from mcp_server.session import ITERATION_CAP, session
from mcp_server.tools.generate_image import generate_image_tool
from mcp_server.tools.inpaint import inpaint_tool


@pytest.fixture(autouse=True)
def fake_pipelines(monkeypatch, tmp_path):
    def fake_generate(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
        return [Image.new("RGB", (8, 8)) for _ in range(num_images)]

    def fake_inpaint(*, image, mask, prompt, negative_prompt, guidance_scale, steps, seed):
        return image.copy()

    import pipeline.generate as generate_module

    monkeypatch.setattr(generate_module, "_default_pipeline_call", fake_generate)
    monkeypatch.setattr(inpaint_module, "_default_pipeline_call", fake_inpaint)
    monkeypatch.setattr("mcp_server.tools._media.OUTPUT_DIR", tmp_path)
    session.clear()
    yield
    session.clear()


def _metadata(result: list) -> dict:
    return next(item for item in result if isinstance(item, dict))


def _images(result: list) -> list[MCPImage]:
    return [item for item in result if isinstance(item, MCPImage)]


def _existing_candidate_path() -> str:
    result = generate_image_tool("a red bicycle", num_images=1)
    return _metadata(result)["candidates"][0]["path"]


def test_inpaint_tool_fixes_the_region_and_returns_image_and_metadata():
    path = _existing_candidate_path()

    result = inpaint_tool(path, x0=0.25, y0=0.25, x1=0.75, y1=0.75, prompt="a normal hand", seed=1)

    images = _images(result)
    metadata = _metadata(result)

    assert len(images) == 1
    assert metadata["ok"] is True
    assert metadata["cap_hit"] is False
    assert metadata["rounds_used"] == 2
    candidate = metadata["candidates"][0]
    assert candidate["prompt"] == "a normal hand"
    assert candidate["source_path"] == path
    assert Path(candidate["path"]).exists()


def test_inpaint_tool_rejects_a_path_outside_the_current_session():
    _existing_candidate_path()

    result = inpaint_tool(
        "outputs/not-in-session.png", x0=0.1, y0=0.1, x1=0.5, y1=0.5, prompt="fix"
    )

    assert result["ok"] is False
    assert result["error"] == "unknown_candidate"


def test_inpaint_tool_rejects_when_no_session_is_active():
    result = inpaint_tool("outputs/whatever.png", x0=0.1, y0=0.1, x1=0.5, y1=0.5, prompt="fix")

    assert result == {
        "ok": False,
        "error": "no_active_session",
        "message": "call generate_image with continue_session=False first",
    }


def test_inpaint_tool_returns_structured_error_for_invalid_bbox():
    path = _existing_candidate_path()

    result = inpaint_tool(path, x0=0.5, y0=0.5, x1=0.1, y1=0.5, prompt="fix")

    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_inpaint_tool_stops_generating_once_the_cap_is_reached():
    path = _existing_candidate_path()

    for _ in range(ITERATION_CAP - 1):
        inpaint_tool(path, x0=0.1, y0=0.1, x1=0.5, y1=0.5, prompt="fix")

    assert session.current().rounds_used == ITERATION_CAP

    result = inpaint_tool(path, x0=0.1, y0=0.1, x1=0.5, y1=0.5, prompt="fix")

    metadata = _metadata(result)
    assert metadata["cap_hit"] is True
    assert metadata["rounds_used"] == ITERATION_CAP
