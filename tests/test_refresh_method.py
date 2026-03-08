import time
import threading
from unittest.mock import MagicMock

from timed_cache import NOT_CACHED, TimedCache, timed_cache

def test_refresh_forces_update_on_fresh_entry() -> None:
    mock = MagicMock(side_effect=lambda *a, **k: mock.call_count)
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    # Initial fetch
    assert cache.get() == 1
    assert mock.call_count == 1

    # Manual refresh
    cache.refresh()
    time.sleep(0.1)  # Give background thread time to run
    assert mock.call_count == 2

    # Verify new value is returned
    assert cache.get() == 2


def test_refresh_on_non_existent_entry() -> None:
    mock = MagicMock(side_effect=lambda *a, **k: mock.call_count)
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    # Refresh non-existent entry
    cache.refresh("new_key")
    time.sleep(0.1)
    assert mock.call_count == 1

    # Verify it was cached
    assert cache.peek("new_key") == 1
    assert cache.get("new_key") == 1


def test_refresh_no_duplicate_tasks() -> None:
    def slow_fetch(*a, **k):
        time.sleep(0.2)
        return 1

    mock = MagicMock(side_effect=slow_fetch)
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    # Trigger first refresh
    cache.refresh()
    # Trigger second refresh immediately
    cache.refresh()

    time.sleep(0.3)
    assert mock.call_count == 1


def test_decorator_expose_refresh() -> None:
    mock = MagicMock(side_effect=lambda x: x + mock.call_count)

    @timed_cache(ttl_seconds=60)
    def cached_fn(x: int) -> int:
        return mock(x)

    assert cached_fn(10) == 11
    assert mock.call_count == 1

    # Use exposed refresh
    cached_fn.refresh(10)
    time.sleep(0.1)
    assert mock.call_count == 2
    assert cached_fn(10) == 12


def test_refresh_sets_ready_event_on_cold_start() -> None:
    # This tests the fix where _background_refresh sets the ready event
    def slow_fetch(*a, **k):
        time.sleep(0.1)
        return "done"

    cache = TimedCache(fetch_fn=slow_fetch)

    # Start background refresh on cold key
    cache.refresh("cold")

    # This should block until background refresh is done, then return "done"
    # If ready event wasn't set, this would hang.
    assert cache.get("cold") == "done"


def test_failed_background_refresh_keeps_stale_value() -> None:
    mock = MagicMock(side_effect=[1, Exception("oops")])
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    # First fetch succeeds
    assert cache.get("k") == 1

    # Manual refresh fails
    cache.refresh("k")
    time.sleep(0.1)

    # Verify stale value 1 is still there
    assert cache.get("k") == 1
    assert mock.call_count == 2


def test_failed_background_refresh_on_cold_key_retries_on_next_get() -> None:
    mock = MagicMock(side_effect=[Exception("oops"), 42])
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    cache.refresh("k")
    time.sleep(0.1)

    assert cache.get("k") == 42
    assert mock.call_count == 2


def test_refresh_is_noop_during_inflight_cold_get() -> None:
    def slow_fetch(*a, **k):
        time.sleep(0.2)
        return 1

    mock = MagicMock(side_effect=slow_fetch)
    cache = TimedCache(fetch_fn=mock, ttl_seconds=60)

    done = threading.Event()

    def worker() -> None:
        assert cache.get("k") == 1
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.05)

    cache.refresh("k")
    done.wait(timeout=1.0)
    thread.join(timeout=1.0)

    assert mock.call_count == 1


def test_failed_cold_background_refresh_handles_replaced_or_missing_entry() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_failing_fetch(*a, **k):
        started.set()
        release.wait(timeout=1.0)
        raise RuntimeError("boom")

    cache = TimedCache(fetch_fn=blocked_failing_fetch, ttl_seconds=60)

    cache.refresh("k")
    assert started.wait(timeout=1.0)

    # Remove the placeholder while background refresh is still in flight so
    # `_entries.get(key) is entry` is false in the failure handler.
    cache.invalidate("k")
    release.set()
    time.sleep(0.1)

    assert cache.peek("k") is NOT_CACHED
    assert cache.size == 0
