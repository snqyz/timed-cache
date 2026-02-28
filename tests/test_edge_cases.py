import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from timed_cache import TimedCache


def test_set_arg_is_hashable_and_order_independent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get({1, 2, 3})
    cache.get({3, 2, 1})
    assert mock.call_count == 1


def test_mutating_list_argument_changes_cache_key(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    payload = [1, 2]
    first = cache.get(payload)

    payload.append(3)
    second = cache.get(payload)

    assert first != second
    assert mock.call_count == 2


def test_negative_ttl_is_effectively_always_stale() -> None:
    mock: MagicMock = MagicMock(side_effect=lambda: mock.call_count)
    cache: TimedCache[int] = TimedCache(fetch_fn=mock, ttl_seconds=-1)

    first = cache.get()
    second = cache.get()

    assert first == 1
    assert second == 1
    time.sleep(0.1)
    assert mock.call_count >= 2


def test_background_refresh_failure_is_logged() -> None:
    call_count = 0

    def flaky_fetch() -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("refresh boom")
        return 10

    cache: TimedCache[int] = TimedCache(fetch_fn=flaky_fetch, ttl_seconds=1)
    cache.get()

    with patch("timed_cache.core.logger.exception") as logger_exception:
        time.sleep(1.1)
        stale = cache.get()
        assert stale == 10
        time.sleep(0.2)

    logger_exception.assert_called_once_with(
        "Background refresh failed for %s; keeping stale value",
        "test_background_refresh_failure_is_logged.<locals>.flaky_fetch",
    )


def test_invalidate_is_idempotent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("x")

    for _ in range(5):
        cache.invalidate("x")

    assert cache.size == 0


def test_invalidate_all_is_idempotent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("a")
    cache.get("b")

    for _ in range(5):
        cache.invalidate_all()

    assert cache.size == 0


def test_background_refresh_does_not_resurrect_invalidated_entry() -> None:
    entered_refresh = threading.Event()
    release_refresh = threading.Event()
    call_count = 0

    def fetch(*args: Any, **kwargs: Any) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            entered_refresh.set()
            release_refresh.wait(timeout=1.0)
        return call_count

    cache: TimedCache[int] = TimedCache(fetch_fn=fetch, ttl_seconds=1)
    cache.get("k")

    time.sleep(1.1)
    stale = cache.get("k")
    assert stale == 1

    assert entered_refresh.wait(timeout=1.0)
    cache.invalidate("k")
    release_refresh.set()
    time.sleep(0.2)

    assert cache.size == 0
    assert cache.get("k") == 3
    assert call_count == 3
