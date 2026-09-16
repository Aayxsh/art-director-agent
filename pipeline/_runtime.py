import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar

T = TypeVar("T")


class PipelineExecutionError(RuntimeError):
    """Raised when an underlying diffusion pipeline call fails or times out."""


def random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def run_with_timeout(fn: Callable[..., T], timeout_seconds: float, **kwargs: object) -> T:
    """Run `fn` in a worker thread, raising PipelineExecutionError on failure or timeout.

    On timeout the worker thread is abandoned, not cancelled — it keeps
    running in the background and holding GPU state after this raises.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        raise PipelineExecutionError(f"pipeline call timed out after {timeout_seconds}s") from exc
    except Exception as exc:
        raise PipelineExecutionError(f"pipeline call failed: {exc}") from exc
    finally:
        executor.shutdown(wait=False)
