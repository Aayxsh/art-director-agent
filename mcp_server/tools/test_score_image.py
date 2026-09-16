import pytest
from PIL import Image

import eval.scoring as scoring_module
import pipeline.generate as generate_module
from mcp_server.session import session
from mcp_server.tools.generate_image import generate_image_tool
from mcp_server.tools.score_image import score_image_tool


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


def _existing_candidate_path() -> str:
    result = generate_image_tool("a red bicycle", num_images=1)
    metadata = next(item for item in result if isinstance(item, dict))
    return metadata["candidates"][0]["path"]


def test_score_image_tool_defaults_to_scoring_against_the_session_brief(monkeypatch):
    path = _existing_candidate_path()
    seen_prompts = []

    def recording_score(*, image, prompt):
        seen_prompts.append(prompt)
        return 77.0

    monkeypatch.setattr(scoring_module, "_default_pipeline_call", recording_score)

    result = score_image_tool(path)

    assert result == {"ok": True, "path": path, "prompt": "a red bicycle", "clip_score": 77.0}
    assert seen_prompts == ["a red bicycle"]


def test_score_image_tool_accepts_an_explicit_prompt():
    path = _existing_candidate_path()

    result = score_image_tool(path, prompt="a different description")

    assert result["prompt"] == "a different description"


def test_score_image_tool_rejects_a_path_outside_the_current_session():
    _existing_candidate_path()

    result = score_image_tool("outputs/not-in-session.png")

    assert result == {
        "ok": False,
        "error": "unknown_candidate",
        "message": "outputs/not-in-session.png is not a candidate from the current session",
    }


def test_score_image_tool_rejects_when_no_session_is_active():
    result = score_image_tool("outputs/whatever.png")

    assert result == {
        "ok": False,
        "error": "no_active_session",
        "message": "call generate_image with continue_session=False first",
    }


def test_score_image_tool_returns_structured_error_when_scoring_fails(monkeypatch):
    path = _existing_candidate_path()

    def failing_score(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scoring_module, "_default_pipeline_call", failing_score)

    result = score_image_tool(path)

    assert result["ok"] is False
    assert result["error"] == "pipeline_failed"
