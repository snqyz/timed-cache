import threading
import time
from collections import defaultdict

import pytest

from timed_cache import TimedCache


def test_background_refresh_respects_max_refresh_workers() -> None:
    active_refreshes = 0
    max_active_refreshes = 0
    refresh_lock = threading.Lock()
    refresh_phase = threading.Event()

    def fetch(key: int) -> int:
        nonlocal active_refreshes, max_active_refreshes
        if refresh_phase.is_set():
            with refresh_lock:
                active_refreshes += 1
                max_active_refreshes = max(max_active_refreshes, active_refreshes)
            time.sleep(0.05)
            with refresh_lock:
                active_refreshes -= 1
        return key

    cache: TimedCache[int] = TimedCache(
        fetch_fn=fetch,
        ttl_seconds=0.01,
        max_refresh_workers=2,
    )

    for key in range(8):
        cache.get(key)

    time.sleep(0.02)
    refresh_phase.set()

    trigger_threads = [
        threading.Thread(target=cache.get, args=(key,)) for key in range(8)
    ]
    for thread in trigger_threads:
        thread.start()
    for thread in trigger_threads:
        thread.join()

    time.sleep(0.4)
    assert max_active_refreshes <= 2


def test_lru_eviction_when_max_entries_reached() -> None:
    fetch_count: dict[str, int] = defaultdict(int)

    def fetch(key: str) -> int:
        fetch_count[key] += 1
        return fetch_count[key]

    cache: TimedCache[int] = TimedCache(fetch_fn=fetch, max_entries=2)

    cache.get("a")
    cache.get("b")
    cache.get("a")  # make "a" most recently used
    cache.get("c")  # should evict "b"

    assert cache.size == 2
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_invalid_eviction_configuration_raises() -> None:
    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        TimedCache(fetch_fn=lambda: 1, max_entries=0)

    with pytest.raises(ValueError, match="max_refresh_workers must be >= 1"):
        TimedCache(fetch_fn=lambda: 1, max_refresh_workers=0)


def test_concurrent_new_keys_do_not_exceed_max_entries() -> None:
    """Even with simultaneous inserts, cache size must stay within capacity."""
    start = threading.Barrier(2)

    def fetch(key: str) -> str:
        time.sleep(0.02)
        return key

    cache: TimedCache[str] = TimedCache(fetch_fn=fetch, max_entries=1)

    def worker(key: str) -> None:
        start.wait()
        cache.get(key)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert cache.size <= 1
