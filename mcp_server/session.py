from dataclasses import dataclass, field

from eval.logging import log_round, start_session_log

# ADR 0003: 5 rounds total, shared across every fix type (reprompt, inpaint,
# param-adjust) — one counter, not a separate budget per tool.
ITERATION_CAP = 5


class NoActiveSession(RuntimeError):
    """Raised when a caller asks to continue a session before one was started."""


@dataclass
class _SessionState:
    brief: str
    log_path: str
    rounds_used: int = 0
    candidates: list[dict] = field(default_factory=list)
    _rounds: list[list[dict]] = field(default_factory=list)


class Session:
    """Server-side session/round-cap tracking (ADR 0003, ADR 0005, ADR 0007).

    A single global session, matching ADR 0002's single-local-GPU,
    single-session-at-a-time scope — no session_id for callers to manage.
    Every round is also persisted to disk (eval/logging.py, ADR 0011) so
    the Phase 5 demo, running as a separate process, can replay it.
    """

    def __init__(self) -> None:
        self._state: _SessionState | None = None

    def start_new(self, brief: str) -> None:
        self._state = _SessionState(brief=brief, log_path=start_session_log(brief))

    def clear(self) -> None:
        self._state = None

    def current(self) -> _SessionState:
        if self._state is None:
            raise NoActiveSession("call generate_image with continue_session=False first")
        return self._state

    def cap_reached(self) -> bool:
        return self.current().rounds_used >= ITERATION_CAP

    def record_round(self, candidates: list[dict]) -> None:
        state = self.current()
        state.rounds_used += 1
        state.candidates.extend(candidates)
        state._rounds.append(candidates)
        log_round(state.log_path, candidates)

    def most_recent_round(self) -> list[dict]:
        state = self.current()
        return state._rounds[-1] if state._rounds else []

    def all_rounds(self) -> list[list[dict]]:
        return list(self.current()._rounds)

    def best_scoring_round(self) -> list[dict]:
        """The round containing the highest-scoring candidate (ADR 0005).

        Falls back to the most recent round if no candidate anywhere in
        the session has a recorded clip_score (e.g. scoring failed for
        everything) — advisory scoring (ADR 0004) never blocks a result.
        """
        state = self.current()
        scored_rounds = [
            r for r in state._rounds if any(c.get("clip_score") is not None for c in r)
        ]
        if not scored_rounds:
            return self.most_recent_round()

        def best_score_in(round_candidates: list[dict]) -> float:
            return max(c["clip_score"] for c in round_candidates if c.get("clip_score") is not None)

        return max(scored_rounds, key=best_score_in)


session = Session()
