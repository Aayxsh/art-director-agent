from pathlib import Path

import pytest
from PIL import Image

import pipeline.generate as generate_module
from mcp_server.tools.generate_image import generate_image_tool


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch, tmp_path):
    def fake_call(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
        return [Image.new("RGB", (8, 8)) for _ in range(num_images)]

    monkeypatch.setattr(generate_module, "_default_pipeline_call", fake_call)
    monkeypatch.setattr("mcp_server.tools.generate_image.OUTPUT_DIR", tmp_path)


def test_generate_image_tool_returns_saved_candidates():
    result = generate_image_tool("a red bicycle", num_images=2, seed=1)

    assert result["ok"] is True
    assert len(result["candidates"]) == 2
    for candidate in result["candidates"]:
        assert candidate["seed"] == 1
        assert candidate["prompt"] == "a red bicycle"
        assert Path(candidate["path"]).exists()


def test_generate_image_tool_returns_structured_error_for_invalid_input():
    result = generate_image_tool("   ")

    assert result["ok"] is False
    assert result["error"] == "invalid_input"
    assert "message" in result


def test_generate_image_tool_returns_structured_error_when_pipeline_fails(monkeypatch):
    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(generate_module, "_default_pipeline_call", failing_call)

    result = generate_image_tool("a red bicycle")

    assert result["ok"] is False
    assert result["error"] == "pipeline_failed"
    assert "message" in result
