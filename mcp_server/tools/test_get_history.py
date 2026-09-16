import pytest
from PIL import Image

import eval.scoring as scoring_module
import pipeline.generate as generate_module
from mcp_server.session import session
from mcp_server.tools.generate_image import generate_image_tool
from mcp_server.tools.get_history import get_history_tool


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch, tmp_path):
    def fake_call(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
        return [Image.new("RGB", (8, 8)) for _ in range(num_images)]

    def fake_score(*, image, prompt):
        return 50.0

    monkeypatch.setattr(generate_module, "_default_pipeline_call", fake_call)
    monkeypatch.setattr(scoring_module, "_default_pipeline_call", fake_score)
    monkeypatch.setattr("mcp_server.tools._media.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("eval.session_log.RESULTS_DIR", tmp_path)
    session.clear()
    yield
    session.clear()


def test_get_history_returns_every_round_of_the_current_session():
    generate_image_tool("a red bicycle", num_images=1, seed=1)
    generate_image_tool("a red bicycle, retry", num_images=1, seed=2, continue_session=True)

    result = get_history_tool()

    assert result["ok"] is True
    assert result["brief"] == "a red bicycle"
    assert result["rounds_used"] == 2
    assert len(result["rounds"]) == 2
    assert result["rounds"][0][0]["seed"] == 1
    assert result["rounds"][1][0]["seed"] == 2


def test_get_history_rejects_when_no_session_is_active():
    result = get_history_tool()

    assert result == {
        "ok": False,
        "error": "no_active_session",
        "message": "call generate_image with continue_session=False first",
    }
