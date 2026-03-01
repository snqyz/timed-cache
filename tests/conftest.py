import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure local module imports work when pytest rootdir is tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timed_cache import TimedCache


@pytest.fixture
def counter_cache_factory() -> Callable[[float], tuple[MagicMock, TimedCache[int]]]:
    """Factory returning a cache whose fetch returns current call count."""

    def _make(
        ttl_seconds: float = 60,
        key_fn: Callable[..., Any] | None = None,
    ) -> tuple[MagicMock, TimedCache[int]]:
        mock = MagicMock(side_effect=lambda *a, **kw: mock.call_count)
        cache: TimedCache[int] = TimedCache(
            fetch_fn=mock,
            ttl_seconds=ttl_seconds,
            key_fn=key_fn,
        )
        return mock, cache

    return _make


@pytest.fixture
def slow_cache_factory() -> Callable[
    [float, float, int],
    tuple[MagicMock, TimedCache[int]],
]:
    """Factory returning a cache with a delayed fetch, useful in concurrency tests."""

    def _make(
        delay: float = 0.1,
        ttl_seconds: float = 60,
        value: int = 42,
        key_fn: Callable[..., Any] | None = None,
    ) -> tuple[MagicMock, TimedCache[int]]:
        import time

        def _slow(*args: Any, **kwargs: Any) -> int:
            time.sleep(delay)
            return value

        mock = MagicMock(side_effect=_slow)
        cache: TimedCache[int] = TimedCache(
            fetch_fn=mock,
            ttl_seconds=ttl_seconds,
            key_fn=key_fn,
        )
        return mock, cache

    return _make
