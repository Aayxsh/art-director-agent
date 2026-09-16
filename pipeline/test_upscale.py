import time

import pytest
from PIL import Image

import pipeline.upscale as upscale_module
from pipeline._runtime import PipelineExecutionError
from pipeline.upscale import InvalidUpscaleInput, upscale


def _fake_pipeline_call(*, image, prompt, negative_prompt, guidance_scale, steps, seed):
    return image.copy()


def _source_image(path, size=(64, 64)):
    img = Image.new("RGB", size, color="blue")
    img.save(path)
    return str(path)


def test_upscale_returns_an_image_scaled_up_by_the_requested_factor(tmp_path):
    source = _source_image(tmp_path / "a.png", size=(100, 100))

    candidate = upscale(
        source,
        prompt="a red bicycle",
        scale=2.0,
        pipeline_call=_fake_pipeline_call,
    )

    assert candidate.image.size == (200, 200)
    assert candidate.source_path == source
    assert candidate.scale == 2.0


def test_upscale_default_scale_is_2x(tmp_path):
    source = _source_image(tmp_path / "a.png", size=(50, 50))

    candidate = upscale(source, prompt="a red bicycle", pipeline_call=_fake_pipeline_call)

    assert candidate.image.size == (100, 100)


def test_upscale_passes_params_through_to_the_pipeline_call(tmp_path):
    source = _source_image(tmp_path / "a.png")
    seen = {}

    def recording_call(*, image, prompt, negative_prompt, guidance_scale, steps, seed):
        seen.update(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )
        return image.copy()

    upscale(
        source,
        prompt="a red bicycle",
        negative_prompt="blurry",
        guidance_scale=9.0,
        steps=12,
        seed=1,
        pipeline_call=recording_call,
    )

    assert seen == {
        "prompt": "a red bicycle",
        "negative_prompt": "blurry",
        "guidance_scale": 9.0,
        "steps": 12,
        "seed": 1,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "   "},
        {"prompt": "x" * 1001},
        {"prompt": "ok", "scale": 0.5},
        {"prompt": "ok", "scale": 5.0},
        {"prompt": "ok", "steps": 0},
        {"prompt": "ok", "steps": 151},
        {"prompt": "ok", "guidance_scale": -1.0},
        {"prompt": "ok", "guidance_scale": 21.0},
    ],
)
def test_upscale_rejects_invalid_input(tmp_path, kwargs):
    source = _source_image(tmp_path / "a.png")
    with pytest.raises(InvalidUpscaleInput):
        upscale(source, pipeline_call=_fake_pipeline_call, **kwargs)


def test_upscale_rejects_a_missing_source_path(tmp_path):
    missing = str(tmp_path / "does-not-exist.png")
    with pytest.raises(InvalidUpscaleInput):
        upscale(missing, prompt="ok", pipeline_call=_fake_pipeline_call)


def test_upscale_wraps_pipeline_failures(tmp_path):
    source = _source_image(tmp_path / "a.png")

    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(PipelineExecutionError):
        upscale(source, prompt="ok", pipeline_call=failing_call)


def test_upscale_wraps_timeouts(tmp_path):
    source = _source_image(tmp_path / "a.png")

    def slow_call(**_kwargs):
        time.sleep(0.2)
        return Image.new("RGB", (64, 64))

    with pytest.raises(PipelineExecutionError):
        upscale(source, prompt="ok", pipeline_call=slow_call, timeout_seconds=0.05)


def test_upscale_never_touches_the_real_pipeline_when_default_call_is_faked(tmp_path, monkeypatch):
    source = _source_image(tmp_path / "a.png")

    def load_real_refiner_must_not_be_called():
        raise AssertionError("_load_real_refiner_pipeline should not run when faked")

    monkeypatch.setattr(
        upscale_module, "_load_real_refiner_pipeline", load_real_refiner_must_not_be_called
    )
    monkeypatch.setattr(upscale_module, "_default_pipeline_call", _fake_pipeline_call)

    candidate = upscale(source, prompt="ok")

    assert candidate.image.size == (128, 128)


@pytest.mark.gpu
def test_upscale_runs_real_sdxl_refiner_end_to_end(tmp_path):
    source = _source_image(tmp_path / "a.png", size=(1024, 1024))

    candidate = upscale(source, prompt="a small blue square", scale=2.0, steps=10)

    assert candidate.image.size == (2048, 2048)
