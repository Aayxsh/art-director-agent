import pytest

from mcp_server.session import ITERATION_CAP, NoActiveSession, session


@pytest.fixture(autouse=True)
def reset_session():
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
