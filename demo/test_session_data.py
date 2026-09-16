import json

from demo.session_data import best_candidate, list_examples, load_example


def _write_example(examples_dir, slug, brief="a red bicycle", rounds=None, upscaled_path=None):
    data = {"brief": brief, "rounds": rounds or [], "upscaled_path": upscaled_path}
    (examples_dir / f"{slug}.json").write_text(json.dumps(data))


def test_list_examples_returns_sorted_slugs(tmp_path):
    _write_example(tmp_path, "02-b")
    _write_example(tmp_path, "01-a")

    assert list_examples(tmp_path) == ["01-a", "02-b"]


def test_list_examples_empty_directory(tmp_path):
    assert list_examples(tmp_path) == []


def test_load_example_returns_brief_rounds_and_upscaled_path(tmp_path):
    _write_example(
        tmp_path,
        "01-a",
        brief="a red bicycle",
        rounds=[[{"path": "images/a.jpg", "seed": 1, "clip_score": 30.0}]],
        upscaled_path="images/a-upscaled.jpg",
    )

    session = load_example(tmp_path, "01-a")

    assert session.slug == "01-a"
    assert session.brief == "a red bicycle"
    assert session.rounds == [[{"path": "images/a.jpg", "seed": 1, "clip_score": 30.0}]]
    assert session.upscaled_path == "images/a-upscaled.jpg"


def test_load_example_upscaled_path_defaults_to_none(tmp_path):
    _write_example(tmp_path, "01-a", rounds=[[{"path": "images/a.jpg", "clip_score": 30.0}]])

    session = load_example(tmp_path, "01-a")

    assert session.upscaled_path is None


def test_best_candidate_returns_the_highest_scoring_across_all_rounds(tmp_path):
    _write_example(
        tmp_path,
        "01-a",
        rounds=[
            [{"path": "images/a.jpg", "clip_score": 30.0}],
            [{"path": "images/b.jpg", "clip_score": 90.0}],
            [{"path": "images/c.jpg", "clip_score": 50.0}],
        ],
    )
    session = load_example(tmp_path, "01-a")

    assert best_candidate(session) == {"path": "images/b.jpg", "clip_score": 90.0}


def test_best_candidate_is_none_for_an_empty_session(tmp_path):
    _write_example(tmp_path, "01-a", rounds=[])
    session = load_example(tmp_path, "01-a")

    assert best_candidate(session) is None
