import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

from timed_cache import TimedCache


def test_fresh_entry_not_refetched(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")
    with patch("timed_cache.time.monotonic", return_value=1059.9):
        cache.get("k")
    assert mock.call_count == 1


def test_stale_entry_triggers_background_refresh(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")

    with patch("timed_cache.time.monotonic", return_value=1061.0):
        stale = cache.get("k")

    assert stale == 1
    time.sleep(0.1)
    assert mock.call_count == 2


def test_after_refresh_new_value_returned(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")

    with patch("timed_cache.time.monotonic", return_value=1061.0):
        cache.get("k")

    time.sleep(0.1)

    with patch("timed_cache.time.monotonic", return_value=1062.0):
        fresh = cache.get("k")

    assert fresh == 2


def test_zero_ttl_always_stale() -> None:
    mock: MagicMock = MagicMock(side_effect=lambda: mock.call_count)
    cache: TimedCache[int] = TimedCache(fetch_fn=mock, ttl_seconds=0)
    cache.get()
    cache.get()
    time.sleep(0.1)
    assert mock.call_count >= 2


def test_custom_ttl_respected(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, _ = counter_cache_factory()
    cache: TimedCache[int] = TimedCache(fetch_fn=mock, ttl_seconds=5)
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")
    with patch("timed_cache.time.monotonic", return_value=1004.9):
        cache.get("k")
    assert mock.call_count == 1


def test_background_refresh_not_spawned_twice(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")

    with patch("timed_cache.time.monotonic", return_value=1061.0):
        cache.get("k")
        cache.get("k")

    time.sleep(0.1)
    assert mock.call_count == 2


def test_stale_read_does_not_block_on_slow_refresh() -> None:
    call_count = 0

    def fetch() -> int:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            time.sleep(0.2)
        return call_count

    cache: TimedCache[int] = TimedCache(fetch_fn=fetch, ttl_seconds=1)
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get()

    with patch("timed_cache.time.monotonic", return_value=1002.0):
        start = time.perf_counter()
        stale = cache.get()
        elapsed = time.perf_counter() - start

    assert stale == 1
    assert elapsed < 0.1
    time.sleep(0.25)
    assert cache.get() == 2


def test_is_refreshing_cleared_after_background_refresh(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    with patch("timed_cache.time.monotonic", return_value=1000.0):
        cache.get("k")
    with patch("timed_cache.time.monotonic", return_value=1061.0):
        cache.get("k")

    time.sleep(0.1)
    key = cache._make_key(("k",), {})
    with cache._lock:
        entry = cache._entries[key]
    assert not entry.is_refreshing


def test_fetched_at_updated_after_refresh(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("k")
    key = cache._make_key(("k",), {})

    with cache._lock:
        first_fetched_at = cache._entries[key].fetched_at

    with patch(
        "timed_cache.time.monotonic",
        return_value=first_fetched_at + cache._ttl_seconds + 1,
    ):
        cache.get("k")

    time.sleep(0.1)
    with cache._lock:
        second_fetched_at = cache._entries[key].fetched_at

    assert second_fetched_at > first_fetched_at
