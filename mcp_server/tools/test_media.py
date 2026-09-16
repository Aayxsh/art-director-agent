from PIL import Image

import eval.scoring as scoring_module
from mcp_server.tools._media import score_or_none


def test_score_or_none_returns_the_score_on_success(monkeypatch):
    monkeypatch.setattr(scoring_module, "_default_pipeline_call", lambda *, image, prompt: 77.0)

    assert score_or_none(Image.new("RGB", (8, 8)), "a red bicycle") == 77.0


def test_score_or_none_returns_none_when_scoring_fails(monkeypatch):
    def failing(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scoring_module, "_default_pipeline_call", failing)

    assert score_or_none(Image.new("RGB", (8, 8)), "a red bicycle") is None
