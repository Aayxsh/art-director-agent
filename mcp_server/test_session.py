import json

import pytest

import eval.logging as logging_module
from mcp_server.session import ITERATION_CAP, NoActiveSession, session


@pytest.fixture(autouse=True)
def reset_session(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_module, "RESULTS_DIR", tmp_path)
    session.clear()
    yield
    session.clear()


def test_start_new_creates_a_fresh_session():
    session.start_new("a red bicycle")

    assert session.current().brief == "a red bicycle"
    assert session.current().rounds_used == 0
    assert session.current().candidates == []


def test_current_raises_when_no_session_started():
    with pytest.raises(NoActiveSession):
        session.current()


def test_record_round_increments_rounds_used_and_appends_candidates():
    session.start_new("a red bicycle")

    session.record_round([{"path": "a.png", "seed": 1}])
    session.record_round([{"path": "b.png", "seed": 2}, {"path": "c.png", "seed": 2}])

    assert session.current().rounds_used == 2
    assert session.current().candidates == [
        {"path": "a.png", "seed": 1},
        {"path": "b.png", "seed": 2},
        {"path": "c.png", "seed": 2},
    ]


def test_cap_not_reached_below_the_limit():
    session.start_new("a red bicycle")
    for _ in range(ITERATION_CAP - 1):
        session.record_round([{"path": "a.png", "seed": 1}])

    assert session.cap_reached() is False


def test_cap_reached_at_the_limit():
    session.start_new("a red bicycle")
    for _ in range(ITERATION_CAP):
        session.record_round([{"path": "a.png", "seed": 1}])

    assert session.cap_reached() is True


def test_most_recent_round_returns_the_last_recorded_batch():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1}])
    session.record_round([{"path": "b.png", "seed": 2}, {"path": "c.png", "seed": 2}])

    assert session.most_recent_round() == [
        {"path": "b.png", "seed": 2},
        {"path": "c.png", "seed": 2},
    ]


def test_most_recent_round_is_empty_before_any_round():
    session.start_new("a red bicycle")

    assert session.most_recent_round() == []


def test_start_new_replaces_an_in_progress_session():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1}])

    session.start_new("a blue scooter")

    assert session.current().brief == "a blue scooter"
    assert session.current().rounds_used == 0


def test_start_new_writes_a_session_log_file():
    session.start_new("a red bicycle")

    data = json.loads(open(session.current().log_path).read())
    assert data["brief"] == "a red bicycle"
    assert data["rounds"] == []


def test_record_round_appends_to_the_session_log_file():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1}])

    data = json.loads(open(session.current().log_path).read())
    assert data["rounds"] == [[{"path": "a.png", "seed": 1}]]


def test_best_scoring_round_returns_the_round_with_the_highest_scoring_candidate():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1, "clip_score": 30.0}])
    session.record_round([{"path": "b.png", "seed": 2, "clip_score": 90.0}])
    session.record_round([{"path": "c.png", "seed": 3, "clip_score": 50.0}])

    assert session.best_scoring_round() == [{"path": "b.png", "seed": 2, "clip_score": 90.0}]


def test_best_scoring_round_falls_back_to_most_recent_when_no_scores_exist():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1}])
    session.record_round([{"path": "b.png", "seed": 2}])

    assert session.best_scoring_round() == [{"path": "b.png", "seed": 2}]


def test_best_scoring_round_is_empty_before_any_round():
    session.start_new("a red bicycle")

    assert session.best_scoring_round() == []


def test_all_rounds_returns_every_round_in_order():
    session.start_new("a red bicycle")
    session.record_round([{"path": "a.png", "seed": 1}])
    session.record_round([{"path": "b.png", "seed": 2}])

    assert session.all_rounds() == [
        [{"path": "a.png", "seed": 1}],
        [{"path": "b.png", "seed": 2}],
    ]
