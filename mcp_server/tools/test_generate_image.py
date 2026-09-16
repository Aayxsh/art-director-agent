from pathlib import Path

import pytest
from mcp.server.mcpserver import Image as MCPImage
from PIL import Image

import eval.scoring as scoring_module
import pipeline.generate as generate_module
from mcp_server.session import ITERATION_CAP, session
from mcp_server.tools.generate_image import generate_image_tool


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch, tmp_path):
    def fake_call(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
        return [Image.new("RGB", (8, 8)) for _ in range(num_images)]

    def fake_score(*, image, prompt):
        return 50.0

    monkeypatch.setattr(generate_module, "_default_pipeline_call", fake_call)
    monkeypatch.setattr(scoring_module, "_default_pipeline_call", fake_score)
    monkeypatch.setattr("mcp_server.tools._media.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("eval.logging.RESULTS_DIR", tmp_path)
    session.clear()
    yield
    session.clear()


def _metadata(result: list) -> dict:
    return next(item for item in result if isinstance(item, dict))


def _images(result: list) -> list[MCPImage]:
    return [item for item in result if isinstance(item, MCPImage)]


def test_generate_image_tool_returns_images_and_saved_metadata():
    result = generate_image_tool("a red bicycle", num_images=2, seed=1)

    images = _images(result)
    metadata = _metadata(result)

    assert len(images) == 2
    for img in images:
        assert img.to_image_content().mime_type == "image/jpeg"
        assert len(img.to_image_content().data) > 0

    assert metadata["ok"] is True
    assert metadata["cap_hit"] is False
    assert metadata["rounds_used"] == 1
    assert len(metadata["candidates"]) == 2
    for candidate in metadata["candidates"]:
        assert candidate["seed"] == 1
        assert candidate["prompt"] == "a red bicycle"
        assert candidate["clip_score"] == 50.0
        assert Path(candidate["path"]).exists()


def test_generate_image_tool_scores_against_the_session_brief_not_the_round_prompt(monkeypatch):
    seen_prompts = []

    def recording_score(*, image, prompt):
        seen_prompts.append(prompt)
        return 50.0

    monkeypatch.setattr(scoring_module, "_default_pipeline_call", recording_score)

    generate_image_tool("a red bicycle", num_images=1)
    generate_image_tool("a red bicycle, fixed hands", num_images=1, continue_session=True)

    assert seen_prompts == ["a red bicycle", "a red bicycle"]


def test_generate_image_tool_omits_clip_score_when_scoring_fails(monkeypatch):
    def failing_score(**_kwargs):
        raise RuntimeError("scoring boom")

    monkeypatch.setattr(scoring_module, "_default_pipeline_call", failing_score)

    result = generate_image_tool("a red bicycle", num_images=1)

    metadata = _metadata(result)
    assert metadata["ok"] is True
    assert metadata["candidates"][0]["clip_score"] is None


def test_generate_image_tool_starts_a_fresh_session_by_default():
    generate_image_tool("a red bicycle", num_images=1)
    generate_image_tool("an unrelated blue scooter", num_images=1)

    assert session.current().brief == "an unrelated blue scooter"
    assert session.current().rounds_used == 1


def test_generate_image_tool_continues_the_session_when_asked():
    generate_image_tool("a red bicycle", num_images=1)
    result = generate_image_tool("a red bicycle, fixed hands", num_images=1, continue_session=True)

    assert session.current().rounds_used == 2
    assert _metadata(result)["rounds_used"] == 2


def test_generate_image_tool_rejects_continue_session_with_no_active_session():
    result = generate_image_tool("a red bicycle", continue_session=True)

    assert result == {
        "ok": False,
        "error": "no_active_session",
        "message": "call generate_image with continue_session=False first",
    }


def test_generate_image_tool_stops_generating_once_the_cap_is_reached(monkeypatch):
    call_count = {"n": 0}

    def counting_call(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
        call_count["n"] += 1
        return [Image.new("RGB", (8, 8)) for _ in range(num_images)]

    # Increasing scores so the most recent round is unambiguously "best" —
    # isolates cap enforcement from best_scoring_round's own tie-breaking.
    score_count = {"n": 0}

    def increasing_score(*, image, prompt):
        score_count["n"] += 1
        return float(score_count["n"])

    monkeypatch.setattr(generate_module, "_default_pipeline_call", counting_call)
    monkeypatch.setattr(scoring_module, "_default_pipeline_call", increasing_score)

    generate_image_tool("a red bicycle", num_images=1)
    for _ in range(ITERATION_CAP - 1):
        generate_image_tool("a red bicycle, retry", num_images=1, continue_session=True)

    assert call_count["n"] == ITERATION_CAP

    result = generate_image_tool("a red bicycle, retry again", num_images=1, continue_session=True)

    assert call_count["n"] == ITERATION_CAP  # no new pipeline call past the cap
    metadata = _metadata(result)
    assert metadata["ok"] is True
    assert metadata["cap_hit"] is True
    assert metadata["rounds_used"] == ITERATION_CAP
    assert len(_images(result)) == 1


def test_generate_image_tool_returns_structured_error_for_invalid_input():
    result = generate_image_tool("   ")

    assert result == {
        "ok": False,
        "error": "invalid_input",
        "message": "prompt must not be empty",
    }


def test_generate_image_tool_returns_structured_error_when_pipeline_fails(monkeypatch):
    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(generate_module, "_default_pipeline_call", failing_call)

    result = generate_image_tool("a red bicycle")

    assert result["ok"] is False
    assert result["error"] == "pipeline_failed"
    assert "message" in result
