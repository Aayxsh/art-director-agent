import time

import pytest
from PIL import Image

import eval.scoring as scoring_module
from eval.scoring import score_image
from pipeline._runtime import PipelineExecutionError


def _fake_pipeline_call(*, image, prompt):
    return 42.0


def test_score_image_returns_the_pipeline_calls_score():
    score = score_image(
        Image.new("RGB", (8, 8)), "a red bicycle", pipeline_call=_fake_pipeline_call
    )

    assert score == 42.0


def test_score_image_passes_image_and_prompt_through():
    seen = {}

    def recording_call(*, image, prompt):
        seen["image"] = image
        seen["prompt"] = prompt
        return 10.0

    img = Image.new("RGB", (8, 8))
    score_image(img, "a red bicycle", pipeline_call=recording_call)

    assert seen["image"] is img
    assert seen["prompt"] == "a red bicycle"


def test_score_image_wraps_pipeline_failures():
    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(PipelineExecutionError):
        score_image(Image.new("RGB", (8, 8)), "a red bicycle", pipeline_call=failing_call)


def test_score_image_wraps_timeouts():
    def slow_call(**_kwargs):
        time.sleep(0.2)
        return 10.0

    with pytest.raises(PipelineExecutionError):
        score_image(
            Image.new("RGB", (8, 8)),
            "a red bicycle",
            pipeline_call=slow_call,
            timeout_seconds=0.05,
        )


def test_score_image_never_touches_the_real_clip_model_when_default_call_is_faked():
    def load_real_clip_must_not_be_called():
        raise AssertionError("_load_real_clip should not run when _default_pipeline_call is faked")

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(scoring_module, "_load_real_clip", load_real_clip_must_not_be_called)
        mp.setattr(scoring_module, "_default_pipeline_call", _fake_pipeline_call)

        score = score_image(Image.new("RGB", (8, 8)), "a red bicycle")

    assert score == 42.0


@pytest.mark.gpu
def test_score_image_runs_real_clip_end_to_end():
    score = score_image(Image.new("RGB", (64, 64), color="red"), "a solid red square")

    assert 0.0 <= score <= 100.0
