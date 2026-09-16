import json

import pytest

import eval.session_log as session_log_module
from eval.session_log import log_round, start_session_log


@pytest.fixture(autouse=True)
def redirect_results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log_module, "RESULTS_DIR", tmp_path)


def test_start_session_log_creates_a_file_with_the_brief():
    path = start_session_log("a red bicycle")

    data = json.loads(open(path).read())
    assert data["brief"] == "a red bicycle"
    assert data["rounds"] == []


def test_log_round_appends_a_round_to_the_session_log():
    path = start_session_log("a red bicycle")

    log_round(path, [{"path": "a.png", "seed": 1, "clip_score": 42.0}])
    log_round(path, [{"path": "b.png", "seed": 2, "clip_score": 55.0}])

    data = json.loads(open(path).read())
    assert data["rounds"] == [
        [{"path": "a.png", "seed": 1, "clip_score": 42.0}],
        [{"path": "b.png", "seed": 2, "clip_score": 55.0}],
    ]


def test_start_session_log_creates_a_distinct_file_per_call():
    path_a = start_session_log("a red bicycle")
    path_b = start_session_log("a blue scooter")

    assert path_a != path_b
