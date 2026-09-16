from dataclasses import dataclass, field

# ADR 0003: 5 rounds total, shared across every fix type (reprompt, inpaint,
# param-adjust) — one counter, not a separate budget per tool.
ITERATION_CAP = 5


class NoActiveSession(RuntimeError):
    """Raised when a caller asks to continue a session before one was started."""


@dataclass
class _SessionState:
    brief: str
    rounds_used: int = 0
    candidates: list[dict] = field(default_factory=list)
    _rounds: list[list[dict]] = field(default_factory=list)


class Session:
    """Server-side session/round-cap tracking (ADR 0003, ADR 0005, Q4).

    A single global session, matching ADR 0002's single-local-GPU,
    single-session-at-a-time scope — no session_id for callers to manage.
    """

    def __init__(self) -> None:
        self._state: _SessionState | None = None

    def start_new(self, brief: str) -> None:
        self._state = _SessionState(brief=brief)

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

    def most_recent_round(self) -> list[dict]:
        state = self.current()
        return state._rounds[-1] if state._rounds else []


session = Session()
