import time

import pytest
from PIL import Image

import pipeline.generate as generate_module
from pipeline.generate import (
    InvalidGenerationInput,
    PipelineExecutionError,
    generate_image,
)


def _fake_pipeline_call(*, prompt, negative_prompt, num_images, guidance_scale, steps, seed):
    return [Image.new("RGB", (8, 8)) for _ in range(num_images)]


def test_generate_image_returns_one_candidate_per_requested_image():
    candidates = generate_image(
        "a red bicycle",
        num_images=2,
        seed=42,
        pipeline_call=_fake_pipeline_call,
    )

    assert len(candidates) == 2
    assert all(c.seed == 42 for c in candidates)
    assert all(c.prompt == "a red bicycle" for c in candidates)
    assert all(isinstance(c.image, Image.Image) for c in candidates)


def test_generate_image_passes_params_through_to_the_pipeline_call():
    seen_calls = []

    def recording_pipeline_call(**kwargs):
        seen_calls.append(kwargs)
        return [Image.new("RGB", (8, 8))]

    generate_image(
        "a red bicycle",
        negative_prompt="blurry",
        num_images=1,
        guidance_scale=9.0,
        steps=25,
        seed=1,
        pipeline_call=recording_pipeline_call,
    )

    assert seen_calls == [
        {
            "prompt": "a red bicycle",
            "negative_prompt": "blurry",
            "num_images": 1,
            "guidance_scale": 9.0,
            "steps": 25,
            "seed": 1,
        }
    ]


def test_generate_image_uses_seed_factory_when_seed_not_given():
    candidates = generate_image(
        "a red bicycle",
        pipeline_call=_fake_pipeline_call,
        seed_factory=lambda: 777,
    )

    assert candidates[0].seed == 777


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "   "},
        {"prompt": "x" * 1001},
        {"prompt": "ok", "num_images": 0},
        {"prompt": "ok", "num_images": 5},
        {"prompt": "ok", "steps": 0},
        {"prompt": "ok", "steps": 151},
        {"prompt": "ok", "guidance_scale": -1.0},
        {"prompt": "ok", "guidance_scale": 21.0},
    ],
)
def test_generate_image_rejects_invalid_input(kwargs):
    prompt = kwargs.pop("prompt")
    with pytest.raises(InvalidGenerationInput):
        generate_image(prompt, pipeline_call=_fake_pipeline_call, **kwargs)


def test_generate_image_wraps_pipeline_failures():
    def failing_pipeline_call(**_kwargs):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(PipelineExecutionError):
        generate_image("a red bicycle", pipeline_call=failing_pipeline_call)


def test_generate_image_wraps_timeouts():
    def slow_pipeline_call(**_kwargs):
        time.sleep(0.2)
        return [Image.new("RGB", (8, 8))]

    with pytest.raises(PipelineExecutionError):
        generate_image(
            "a red bicycle",
            pipeline_call=slow_pipeline_call,
            timeout_seconds=0.05,
        )


@pytest.mark.gpu
def test_generate_image_runs_real_sdxl_pipeline_end_to_end():
    candidates = generate_image("a small red bicycle on a white background", num_images=1, steps=15)

    assert len(candidates) == 1
    assert candidates[0].image.size[0] > 0


@pytest.mark.gpu
def test_generate_image_does_not_count_pipeline_warm_up_against_the_timeout(monkeypatch):
    class FakeResult:
        images = [Image.new("RGB", (8, 8))]

    class FakePipe:
        def __call__(self, **_kwargs):
            return FakeResult()

    load_count = {"n": 0}

    def slow_first_time_load():
        load_count["n"] += 1
        if load_count["n"] == 1:
            time.sleep(0.2)
        return FakePipe()

    monkeypatch.setattr(generate_module, "_real_pipeline", None)
    monkeypatch.setattr(generate_module, "_load_real_pipeline", slow_first_time_load)

    candidates = generate_image("a red bicycle", timeout_seconds=0.05)

    assert len(candidates) == 1
    assert candidates[0].image.size[0] > 0
