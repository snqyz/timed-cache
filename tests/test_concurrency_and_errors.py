import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from timed_cache import TimedCache


def test_concurrent_cold_requests_never_return_none(
    slow_cache_factory: Callable[
        [float, float, int],
        tuple[MagicMock, TimedCache[int]],
    ],
) -> None:
    _, cache = slow_cache_factory(delay=0.15)
    results: list[int | None] = []
    barrier = threading.Barrier(5)

    def worker() -> None:
        barrier.wait()
        results.append(cache.get("k"))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(value is not None for value in results)
    assert all(value == 42 for value in results)


def test_concurrent_cold_requests_only_fetch_once(
    slow_cache_factory: Callable[
        [float, float, int],
        tuple[MagicMock, TimedCache[int]],
    ],
) -> None:
    mock, cache = slow_cache_factory(delay=0.1)
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        cache.get("k")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert mock.call_count == 1


def test_concurrent_different_keys_fetch_independently(
    slow_cache_factory: Callable[
        [float, float, int],
        tuple[MagicMock, TimedCache[int]],
    ],
) -> None:
    mock, cache = slow_cache_factory(delay=0.05)
    barrier = threading.Barrier(4)
    keys = ["a", "b", "c", "d"]
    results: dict[str, int] = {}

    def worker(key: str) -> None:
        barrier.wait()
        results[key] = cache.get(key)

    threads = [threading.Thread(target=worker, args=(key,)) for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert mock.call_count == 4
    assert len(results) == 4


def test_cold_fetch_error_propagates() -> None:
    def bad_fetch() -> int:
        raise ValueError("fetch failed")

    cache: TimedCache[int] = TimedCache(fetch_fn=bad_fetch)
    with pytest.raises(ValueError, match="fetch failed"):
        cache.get()


def test_failed_cold_fetch_removes_entry() -> None:
    call_count = 0

    def flaky() -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first call fails")
        return 99

    cache: TimedCache[int] = TimedCache(fetch_fn=flaky)
    with pytest.raises(RuntimeError):
        cache.get()
    assert cache.size == 0
    assert cache.get() == 99


def test_concurrent_cold_fetch_error_unblocks_waiting_threads() -> None:
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def failing_fetch() -> int:
        time.sleep(0.05)
        raise RuntimeError("boom")

    cache: TimedCache[int] = TimedCache(fetch_fn=failing_fetch)

    def worker() -> None:
        barrier.wait()
        try:
            cache.get()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert len(errors) == 3
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all(str(error) == "boom" for error in errors)


def test_background_refresh_error_preserves_stale_value() -> None:
    call_count = 0

    def sometimes_fail(key: str) -> int:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("refresh failed")
        return 7

    cache: TimedCache[int] = TimedCache(fetch_fn=sometimes_fail, ttl_seconds=1)
    cache.get("k")

    time.sleep(1.1)
    stale = cache.get("k")
    time.sleep(0.2)

    assert stale == 7


def test_background_refresh_error_allows_retry_next_call() -> None:
    call_count = 0
    refreshed = threading.Event()

    def sometimes_fail() -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("transient error")
        if call_count == 3:
            refreshed.set()
        return call_count

    cache: TimedCache[int] = TimedCache(fetch_fn=sometimes_fail, ttl_seconds=1)
    cache.get()
    time.sleep(1.1)
    cache.get()
    time.sleep(0.2)

    time.sleep(1.0)
    cache.get()
    refreshed.wait(timeout=1.0)

    assert refreshed.is_set()
    assert call_count == 3


def test_hammer_single_key_never_returns_none() -> None:
    def fetch() -> str:
        time.sleep(0.01)
        return "ok"

    cache: TimedCache[str] = TimedCache(fetch_fn=fetch, ttl_seconds=0.05)
    results: list[str | None] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            value = cache.get()
            with lock:
                results.append(value)
            time.sleep(0.01)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(value is not None for value in results)
    assert all(value == "ok" for value in results)


def test_hammer_many_keys_no_exceptions() -> None:
    def fetch(key: int) -> int:
        return key * 2

    cache: TimedCache[int] = TimedCache(fetch_fn=fetch, ttl_seconds=0.1)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for i in range(50):
                cache.get(i % 10)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_invalidate_under_concurrent_access(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    errors: list[Exception] = []

    def reader() -> None:
        try:
            for _ in range(50):
                cache.get("k")
                time.sleep(0.001)
        except Exception as error:
            errors.append(error)

    def invalidator() -> None:
        try:
            for _ in range(20):
                cache.invalidate("k")
                time.sleep(0.003)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=reader) for _ in range(5)] + [
        threading.Thread(target=invalidator) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_ready_event_set_after_cold_fetch(
    slow_cache_factory: Callable[
        [float, float, int],
        tuple[MagicMock, TimedCache[int]],
    ],
) -> None:
    _, cache = slow_cache_factory(delay=0.05)
    thread = threading.Thread(target=cache.get, args=("k",))
    thread.start()
    thread.join()

    key = cache._key_fn("k",)
    entry = cache._entries[key]
    assert entry.ready.is_set()


def test_invalidate_during_inflight_cold_fetch_forces_refetch() -> None:
    entered = threading.Event()
    release = threading.Event()
    call_count = 0

    def slow_fetch(*args: Any, **kwargs: Any) -> int:
        nonlocal call_count
        call_count += 1
        entered.set()
        release.wait(timeout=1.0)
        return call_count

    cache: TimedCache[int] = TimedCache(fetch_fn=slow_fetch)
    thread = threading.Thread(target=cache.get, args=("k",))
    thread.start()
    assert entered.wait(timeout=1.0)

    cache.invalidate("k")
    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    assert cache.get("k") == 2
    assert call_count == 2


def test_size_counts_placeholder_during_cold_fetch() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_fetch(*args: Any, **kwargs: Any) -> int:
        entered.set()
        release.wait(timeout=1.0)
        return 10

    cache: TimedCache[int] = TimedCache(fetch_fn=slow_fetch)
    thread = threading.Thread(target=cache.get, args=("k",))
    thread.start()
    assert entered.wait(timeout=1.0)

    assert cache.size == 1
    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_peek_during_inflight_cold_fetch_returns_none() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_fetch(*args: Any, **kwargs: Any) -> int:
        entered.set()
        release.wait(timeout=1.0)
        return 10

    cache: TimedCache[int] = TimedCache(fetch_fn=slow_fetch)
    thread = threading.Thread(target=cache.get, args=("k",))
    thread.start()
    assert entered.wait(timeout=1.0)

    assert cache.peek("k") is None
    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
