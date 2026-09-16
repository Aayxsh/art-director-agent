import time

import pytest
from PIL import Image

import pipeline.inpaint as inpaint_module
from pipeline._runtime import PipelineExecutionError
from pipeline.inpaint import InvalidInpaintInput, inpaint


def _fake_pipeline_call(*, image, mask, prompt, negative_prompt, guidance_scale, steps, seed):
    return image.copy()


def _source_image(path, size=(64, 64)):
    img = Image.new("RGB", size, color="red")
    img.save(path)
    return str(path)


def test_inpaint_returns_a_candidate_for_the_masked_region(tmp_path):
    source = _source_image(tmp_path / "a.png")

    candidate = inpaint(
        source,
        bbox=(0.25, 0.25, 0.75, 0.75),
        prompt="a normal hand",
        seed=1,
        pipeline_call=_fake_pipeline_call,
    )

    assert candidate.seed == 1
    assert candidate.prompt == "a normal hand"
    assert candidate.source_path == source
    assert candidate.bbox == (0.25, 0.25, 0.75, 0.75)
    assert isinstance(candidate.image, Image.Image)


def test_inpaint_passes_a_correctly_sized_binary_mask_to_the_pipeline_call(tmp_path):
    source = _source_image(tmp_path / "a.png", size=(100, 100))
    seen = {}

    def recording_call(*, image, mask, prompt, negative_prompt, guidance_scale, steps, seed):
        seen["mask"] = mask
        seen["image_size"] = image.size
        return image.copy()

    inpaint(
        source,
        bbox=(0.1, 0.2, 0.5, 0.6),
        prompt="fix",
        seed=1,
        pipeline_call=recording_call,
    )

    mask = seen["mask"]
    assert mask.size == seen["image_size"] == (100, 100)
    assert mask.mode == "L"
    # inside the bbox is white (inpaint), outside is black (keep)
    assert mask.getpixel((30, 40)) == 255
    assert mask.getpixel((5, 5)) == 0


@pytest.mark.parametrize(
    "bbox",
    [
        (0.5, 0.5, 0.5, 0.6),  # x0 == x1
        (0.5, 0.5, 0.4, 0.6),  # x0 > x1
        (0.1, 0.5, 0.5, 0.5),  # y0 == y1
        (-0.1, 0.1, 0.5, 0.5),  # out of range
        (0.1, 0.1, 1.5, 0.5),  # out of range
    ],
)
def test_inpaint_rejects_invalid_bbox(tmp_path, bbox):
    source = _source_image(tmp_path / "a.png")

    with pytest.raises(InvalidInpaintInput):
        inpaint(source, bbox=bbox, prompt="fix", pipeline_call=_fake_pipeline_call)


def test_inpaint_rejects_empty_prompt(tmp_path):
    source = _source_image(tmp_path / "a.png")

    with pytest.raises(InvalidInpaintInput):
        inpaint(source, bbox=(0.1, 0.1, 0.5, 0.5), prompt="  ", pipeline_call=_fake_pipeline_call)


def test_inpaint_rejects_a_missing_source_path(tmp_path):
    missing = str(tmp_path / "does-not-exist.png")

    with pytest.raises(InvalidInpaintInput):
        inpaint(missing, bbox=(0.1, 0.1, 0.5, 0.5), prompt="fix", pipeline_call=_fake_pipeline_call)


def test_inpaint_wraps_pipeline_failures(tmp_path):
    source = _source_image(tmp_path / "a.png")

    def failing_call(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(PipelineExecutionError):
        inpaint(source, bbox=(0.1, 0.1, 0.5, 0.5), prompt="fix", pipeline_call=failing_call)


def test_inpaint_wraps_timeouts(tmp_path):
    source = _source_image(tmp_path / "a.png")

    def slow_call(**_kwargs):
        time.sleep(0.2)
        return Image.new("RGB", (64, 64))

    with pytest.raises(PipelineExecutionError):
        inpaint(
            source,
            bbox=(0.1, 0.1, 0.5, 0.5),
            prompt="fix",
            pipeline_call=slow_call,
            timeout_seconds=0.05,
        )


def test_inpaint_never_touches_the_real_pipeline_when_default_call_is_faked(tmp_path, monkeypatch):
    source = _source_image(tmp_path / "a.png")

    def load_real_pipeline_must_not_be_called():
        raise AssertionError(
            "_load_real_pipeline should not run when _default_pipeline_call is faked"
        )

    monkeypatch.setattr(
        inpaint_module, "_load_real_pipeline", load_real_pipeline_must_not_be_called
    )
    monkeypatch.setattr(inpaint_module, "_default_pipeline_call", _fake_pipeline_call)

    candidate = inpaint(source, bbox=(0.1, 0.1, 0.5, 0.5), prompt="fix")

    assert isinstance(candidate.image, Image.Image)


@pytest.mark.gpu
def test_inpaint_runs_real_sdxl_inpainting_pipeline_end_to_end(tmp_path):
    source = _source_image(tmp_path / "a.png", size=(1024, 1024))

    candidate = inpaint(
        source,
        bbox=(0.3, 0.3, 0.7, 0.7),
        prompt="a small blue square",
        steps=15,
    )

    assert candidate.image.size == (1024, 1024)
