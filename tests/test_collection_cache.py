import threading
import time
from unittest.mock import MagicMock

import pytest

from timed_cache import TimedCollection


def test_collection_batch_fetches_on_cold_start():
    mock = MagicMock(side_effect=lambda keys: {k: len(k) for k in keys})
    cache = TimedCollection(fetch_fn=mock)

    results = cache.get_collection(["apple", "banana"])
    assert results == {"apple": 5, "banana": 6}
    assert mock.call_count == 1
    mock.assert_called_once_with(["apple", "banana"])


def test_collection_supports_getitem():
    cache = TimedCollection(fetch_fn=lambda keys: {k: len(k) for k in keys})
    cache.get_collection(["a", "bc"])

    assert cache["a"] == 1
    assert cache["bc"] == 2


def test_collection_individual_ttl():
    mock = MagicMock(side_effect=lambda keys: {k: time.monotonic() for k in keys})
    # Set TTL to 0.5s
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.5)

    # Initial fetch for 'a'
    t1 = cache.get_collection(["a"])["a"]
    assert mock.call_count == 1

    time.sleep(0.2)
    # Fetch 'b' (cold) and 'a' (fresh)
    res = cache.get_collection(["a", "b"])
    assert res["a"] == t1
    assert "b" in res
    assert mock.call_count == 2  # one for 'a', one for 'b' (cold)

    time.sleep(0.4)
    # Now 'a' is stale (> 0.5s), but 'b' is still fresh (only 0.4s old)
    res = cache.get_collection(["a", "b"])
    assert res["a"] == t1  # returns stale value immediately

    # Wait for the background refresh to complete (to be 3)
    for _ in range(20):
        if mock.call_count == 3:
            break
        time.sleep(0.05)
    assert mock.call_count == 3

    # Verify 'a' was refreshed
    assert cache["a"] > t1


def test_collection_single_key_access_uses_batch_fn():
    mock = MagicMock(side_effect=lambda keys: {k: len(k) for k in keys})
    cache = TimedCollection(fetch_fn=mock)

    assert cache["word"] == 4
    mock.assert_called_once_with(["word"])


def test_collection_batch_refresh():
    mock = MagicMock(side_effect=lambda keys: {k: time.monotonic() for k in keys})
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.1)

    cache.get_collection(["a", "b"])
    assert mock.call_count == 1

    time.sleep(0.2)  # both stale
    res = cache.get_collection(["a", "b"])
    # Should trigger ONE batch refresh for both
    for _ in range(20):
        if mock.call_count == 2:
            break
        time.sleep(0.05)
    assert mock.call_count == 2
    mock.assert_called_with(["a", "b"])


def test_fetch_single_missing_key():
    cache = TimedCollection(fetch_fn=lambda keys: {})
    with pytest.raises(KeyError, match="missing from fetch_fn results"):
        _ = cache["missing"]


def test_get_collection_already_refreshing():
    # Covers line 374: if not entry.is_refreshing
    mock = MagicMock(side_effect=lambda keys: dict.fromkeys(keys, 1))
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.1)

    cache.get_collection(["a"])
    time.sleep(0.2)

    # Manually set is_refreshing to True for 'a'
    ckey = cache._key_fn("a")
    with cache._lock:
        cache._entries[ckey].is_refreshing = True

    # Now call get_collection. It should NOT add 'a' to stale_keys
    cache.get_collection(["a"])
    # If it didn't add to stale_keys, _batch_refresh wasn't submitted (or at least no new refresh started here)
    # Actually TimedCache.get might have started it if we used cache.get, but get_collection has its own logic.

    # Check that no new call was made (only the initial one)
    assert mock.call_count == 1


def test_get_collection_entry_exists_after_fetch():
    # Covers line 395-397: else block where entry exists after fetch
    # This happens if another thread inserted the entry while we were fetching.
    def slow_fetch(keys):
        time.sleep(0.2)
        return dict.fromkeys(keys, 1)

    cache = TimedCollection(fetch_fn=slow_fetch)

    import threading

    def target():
        cache.get_collection(["a"])

    t = threading.Thread(target=target)
    t.start()

    time.sleep(0.1)
    # Manually insert entry for 'a'
    ckey = cache._key_fn("a")
    with cache._lock:
        entry = _CacheEntry(value=2, fetched_at=time.monotonic())
        entry.ready.set()
        cache._entries[ckey] = entry

    t.join()
    # The slow_fetch should have finished and hit the 'else' block
    # and updated the entry to value=1
    assert cache["a"] == 1


def test_batch_refresh_failure(caplog):
    # Covers lines 420-431
    mock = MagicMock(side_effect=[{"a": 1}, Exception("Fetch failed")])
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.1)

    cache.get_collection(["a"])
    assert mock.call_count == 1
    time.sleep(0.2)

    import logging

    with caplog.at_level(logging.ERROR):
        cache.get_collection(["a"])
        # Wait for background refresh to fail
        for _ in range(20):
            if "Background batch refresh failed" in caplog.text:
                break
            time.sleep(0.05)

    assert "Background batch refresh failed" in caplog.text
    # Stale value should still be there
    assert cache["a"] == 1


def test_batch_refresh_entry_not_found():
    # Covers line 414: if entry is None in _batch_refresh
    # This can happen if an entry is invalidated while being refreshed
    def slow_fetch(keys):
        time.sleep(0.2)
        return dict.fromkeys(keys, 1)

    cache = TimedCollection(fetch_fn=slow_fetch, ttl_seconds=0.1)

    cache.get_collection(["a"])
    time.sleep(0.2)

    # Trigger refresh
    cache.get_collection(["a"])
    # Invalidate 'a' immediately while it is refreshing
    cache.invalidate("a")

    # Wait for background refresh to finish
    time.sleep(0.3)
    # If it didn't crash, good.
    assert cache.peek("a") is None


def test_batch_refresh_failure_cold_entry(caplog):
    # Covers line 428-430: if not entry.ready.is_set()
    # To trigger this, we need a cold entry to fail during _batch_refresh.
    # While get_collection doesn't currently do this, _batch_refresh should handle it.
    mock = MagicMock(side_effect=Exception("Fetch failed"))
    cache = TimedCollection(fetch_fn=mock)

    ckey = cache._key_fn("a")
    entry = _CacheEntry(value=None, fetched_at=None)
    # entry.ready is NOT set
    with cache._lock:
        cache._entries[ckey] = entry

    import logging

    with caplog.at_level(logging.ERROR):
        # Manually call _batch_refresh
        cache._batch_refresh(["a"], {})

    assert "Background batch refresh failed" in caplog.text
    # Entry should have been removed
    assert cache.peek("a") is None
    assert ckey not in cache._entries


def test_batch_refresh_failure_entry_missing(caplog):
    # Covers line 426: if entry is None in exception handler
    mock = MagicMock(side_effect=Exception("Fetch failed"))
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.1)

    # Pre-populate
    with cache._lock:
        ckey = cache._key_fn("a")
        entry = _CacheEntry(value=1, fetched_at=time.monotonic())
        entry.ready.set()
        cache._entries[ckey] = entry

    import logging

    with caplog.at_level(logging.ERROR):
        # Manually call _batch_refresh with two keys, one missing
        cache._batch_refresh(["a", "missing"], {})

    assert "Background batch refresh failed" in caplog.text
    # 'a' should still be there (stale)
    assert cache["a"] == 1


from timed_cache.core import _CacheEntry


def test_get_collection_waits_for_inflight_cold_fetch():
    call_count = 0
    lock = threading.Lock()

    def slow_fetch(keys):
        nonlocal call_count
        with lock:
            call_count += 1
        time.sleep(0.2)
        return dict.fromkeys(keys, 1)

    cache = TimedCollection(fetch_fn=slow_fetch)
    first_result: dict[str, int] = {}

    def first_call():
        first_result.update(cache.get_collection(["a"]))

    t = threading.Thread(target=first_call)
    t.start()
    time.sleep(0.05)
    second_result = cache.get_collection(["a"])
    t.join()

    assert first_result == {"a": 1}
    assert second_result == {"a": 1}
    assert call_count == 1


def test_get_collection_missing_cold_key_raises_and_allows_retry():
    mock = MagicMock(side_effect=[{}, {"a": 1}])
    cache = TimedCollection(fetch_fn=mock)

    with pytest.raises(KeyError, match="Keys missing from fetch_fn results"):
        cache.get_collection(["a"])

    assert cache.peek("a") is None
    assert cache.get_collection(["a"]) == {"a": 1}
    assert mock.call_count == 2


def test_get_collection_rolls_back_refresh_flag_when_submit_fails():
    cache = TimedCollection(
        fetch_fn=lambda keys: dict.fromkeys(keys, 1), ttl_seconds=0.1
    )
    cache.get_collection(["a"])
    time.sleep(0.2)

    original_submit = cache._refresh_executor.submit

    def fail_submit(*args, **kwargs):
        raise RuntimeError("submit failed")

    cache._refresh_executor.submit = fail_submit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="submit failed"):
        cache.get_collection(["a"])

    ckey = cache._key_fn("a")
    with cache._lock:
        assert cache._entries[ckey].is_refreshing is False

    cache._refresh_executor.submit = original_submit  # type: ignore[method-assign]


def test_batch_refresh_missing_key_clears_refreshing_flag():
    mock = MagicMock(side_effect=[{"a": 1, "b": 2}, {"a": 3}, {"b": 4}])
    cache = TimedCollection(fetch_fn=mock, ttl_seconds=0.1)

    cache.get_collection(["a", "b"])
    time.sleep(0.2)
    cache.get_collection(["a", "b"])

    for _ in range(20):
        if mock.call_count >= 2:
            break
        time.sleep(0.05)
    assert mock.call_count == 2

    ckey_b = cache._key_fn("b")
    with cache._lock:
        assert cache._entries[ckey_b].is_refreshing is False

    time.sleep(0.2)
    cache.get_collection(["b"])
    for _ in range(20):
        if mock.call_count >= 3:
            break
        time.sleep(0.05)
    assert mock.call_count == 3


def test_get_collection_duplicate_key_after_eviction_skips_duplicate_fetch():
    calls: list[list[str]] = []

    def fetch(keys):
        calls.append(list(keys))
        return {k: len(k) for k in keys}

    cache = TimedCollection(fetch_fn=fetch, max_entries=1)
    result = cache.get_collection(["a", "bb", "a"])

    assert result == {"a": 1, "bb": 2}
    # "a" appears once in the cold batch even though it appears twice in input.
    assert calls == [["a", "bb"]]


def test_get_collection_cold_fetch_exception_cleans_placeholders():
    cache = TimedCollection(
        fetch_fn=lambda keys: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        cache.get_collection(["a", "b"])

    assert cache.peek("a") is None
    assert cache.peek("b") is None


def test_get_collection_cold_fetch_exception_handles_evicted_placeholder():
    # max_entries=1 evicts the first placeholder while scanning cold keys.
    cache = TimedCollection(
        fetch_fn=lambda keys: (_ for _ in ()).throw(RuntimeError("boom")),
        max_entries=1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        cache.get_collection(["a", "b"])

    assert cache.peek("a") is None
    assert cache.peek("b") is None


def test_get_collection_entry_none_path_after_cold_fetch():
    gate = threading.Event()
    release = threading.Event()

    def fetch(keys):
        gate.set()
        release.wait(timeout=1.0)
        return {"a": 1, "b": 2}

    cache = TimedCollection(fetch_fn=fetch)
    result_holder: dict[str, int] = {}

    def run():
        result_holder.update(cache.get_collection(["a", "b"]))

    t = threading.Thread(target=run)
    t.start()
    gate.wait(timeout=1.0)

    # Remove key "b" while the fetch is in progress; update loop will see entry is None.
    cache.invalidate("b")
    release.set()
    t.join()

    assert result_holder == {"a": 1, "b": 2}


def test_get_collection_missing_key_keeps_ready_entry_inserted_during_fetch():
    gate = threading.Event()
    release = threading.Event()

    def fetch(keys):
        gate.set()
        release.wait(timeout=1.0)
        return {}

    cache = TimedCollection(fetch_fn=fetch)
    thread_error: list[Exception] = []

    def run():
        try:
            cache.get_collection(["a"])
        except Exception as error:  # pragma: no cover - asserted below
            thread_error.append(error)

    t = threading.Thread(target=run)
    t.start()
    gate.wait(timeout=1.0)

    ckey = cache._key_fn("a")
    with cache._lock:
        entry = cache._entries[ckey]
        entry.value = 123
        entry.fetched_at = time.monotonic()
        entry.ready.set()

    release.set()
    t.join()

    assert thread_error
    assert isinstance(thread_error[0], KeyError)
    assert cache.peek("a") == 123


def test_get_collection_waiting_entry_raises_error():
    gate = threading.Event()
    release = threading.Event()

    def failing_fetch(keys):
        gate.set()
        release.wait(timeout=1.0)
        raise ValueError("fail later")

    cache = TimedCollection(fetch_fn=failing_fetch)
    thread_errors: list[Exception] = []

    def owner():
        try:
            cache.get_collection(["a"])
        except Exception as error:  # pragma: no cover - asserted below
            thread_errors.append(error)

    t = threading.Thread(target=owner)
    t.start()
    gate.wait(timeout=1.0)

    with pytest.raises(ValueError, match="fail later"):
        cache.get_collection(["a"])

    release.set()
    t.join()
    assert thread_errors
    assert isinstance(thread_errors[0], ValueError)


def test_get_collection_submit_failure_with_missing_entry_branch():
    cache = TimedCollection(
        fetch_fn=lambda keys: dict.fromkeys(keys, 1), ttl_seconds=0.1
    )
    cache.get_collection(["a"])
    time.sleep(0.2)

    def fail_submit(*args, **kwargs):
        cache.invalidate("a")
        raise RuntimeError("submit failed")

    cache._refresh_executor.submit = fail_submit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="submit failed"):
        cache.get_collection(["a"])

    assert cache.peek("a") is None


def test_batch_refresh_missing_key_for_not_ready_entry_removes_it():
    cache = TimedCollection(fetch_fn=lambda keys: {})
    ckey = cache._key_fn("a")
    with cache._lock:
        entry = _CacheEntry(value=None, fetched_at=None)
        cache._entries[ckey] = entry

    cache._batch_refresh(["a"], {})

    assert cache.peek("a") is None
    with cache._lock:
        assert ckey not in cache._entries


def test_peek_collection():
    mock = MagicMock(side_effect=lambda keys: {k: len(k) for k in keys})
    cache = TimedCollection(fetch_fn=mock)

    # Initial state: empty
    assert cache.peek_collection(["a", "b"]) == {}

    # Fetch "a"
    cache.get_collection(["a"])
    assert cache.peek_collection(["a", "b"]) == {"a": 1}

    # Fetch "b"
    cache.get_collection(["b"])
    assert cache.peek_collection(["a", "b"]) == {"a": 1, "b": 1}

    # Verify peek doesn't trigger fetch
    mock.reset_mock()
    assert cache.peek_collection(["c"]) == {}
    mock.assert_not_called()


def test_invalidate_collection():
    cache = TimedCollection(fetch_fn=lambda keys: {k: len(k) for k in keys})

    cache.get_collection(["a", "b", "c"])
    assert cache.peek_collection(["a", "b", "c"]) == {"a": 1, "b": 1, "c": 1}

    cache.invalidate_collection(["a", "c"])
    assert cache.peek_collection(["a", "b", "c"]) == {"b": 1}

    # Invalidating non-existent key should not raise
    cache.invalidate_collection(["non-existent"])


def test_peek_collection_with_kwargs():
    def fetch_with_tag(keys, tag="none"):
        return {k: f"{k}-{tag}" for k in keys}

    cache = TimedCollection(fetch_fn=fetch_with_tag)

    cache.get_collection(["a"], tag="foo")

    # Peek with same tag should find it
    assert cache.peek_collection(["a"], tag="foo") == {"a": "a-foo"}

    # Peek with different tag should NOT find it (different cache key)
    assert cache.peek_collection(["a"], tag="bar") == {}


def test_invalidate_collection_with_kwargs():
    def fetch_with_tag(keys, tag="none"):
        return {k: f"{k}-{tag}" for k in keys}

    cache = TimedCollection(fetch_fn=fetch_with_tag)

    cache.get_collection(["a"], tag="foo")
    cache.get_collection(["a"], tag="bar")

    assert cache.peek_collection(["a"], tag="foo") == {"a": "a-foo"}
    assert cache.peek_collection(["a"], tag="bar") == {"a": "a-bar"}

    # Invalidate one tag
    cache.invalidate_collection(["a"], tag="foo")

    assert cache.peek_collection(["a"], tag="foo") == {}
    assert cache.peek_collection(["a"], tag="bar") == {"a": "a-bar"}
