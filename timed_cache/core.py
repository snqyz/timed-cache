"""Thread-safe timed cache with stale-while-revalidate semantics.

Concurrency model at a glance:
- Exactly one thread owns an initial (cold) fetch for a given key.
- Other threads targeting the same cold key wait on that key's ``ready`` event.
- Once a value exists, stale reads return immediately and trigger background
  refresh at most once per key at a time.
- Locks protect only shared in-memory state; fetch functions always run outside
  the lock to avoid global contention and lock convoying.
"""

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, cast, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
P = ParamSpec("P")
R = TypeVar("R")

# A cache key is built from the positional and keyword arguments passed to get().
CacheKey = tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]


@dataclass
class _CacheEntry(Generic[T]):
    """A single cached value.

    ``ready`` is unset while the initial fetch is in progress; all threads
    that arrive for the same cold key wait on it before reading ``value``.
    For warm or stale entries it is always set, so ``wait()`` returns
    immediately.
    """

    value: T | None
    fetched_at: float | None
    error: Exception | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    is_refreshing: bool = False


class TimedCache(Generic[T]):
    """Wraps a fetch function and caches its result per unique set of arguments.

    Each distinct combination of arguments gets its own cache entry with an
    independent TTL (default: 5 minutes).  When an entry becomes stale the
    *next* caller receives the stale value immediately and a background thread
    silently refreshes it; no caller ever blocks on a refresh.

    Key Generation:
    By default, this uses the fast ``make_key`` which assumes all arguments are
    hashable (e.g., int, str, tuple). If you need to pass mutable arguments
    (e.g., list, dict, set), provide ``key_fn=TimedCache.deep_key_fn`` to the
    constructor.

    Errors during a cold-start fetch propagate to the caller.  Errors during a
    background refresh are logged and the stale value is kept so that the next
    call can try again.

    Usage::

        def fetch_user(user_id: int, *, include_profile: bool = False) -> dict:
            ...

        cache: TimedCache[dict] = TimedCache(fetch_fn=fetch_user)
        user = cache.get(42, include_profile=True)
    """

    def __init__(
        self,
        fetch_fn: Callable[..., T],
        ttl_seconds: float = 300,
        max_entries: int | None = None,
        max_refresh_workers: int = 8,
        key_fn: Callable[..., Any] | None = None,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 when provided")
        if max_refresh_workers < 1:
            raise ValueError("max_refresh_workers must be >= 1")

        self._fetch_fn = fetch_fn
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._key_fn = key_fn or self.make_key
        self._entries: OrderedDict[Any, _CacheEntry[T]] = OrderedDict()
        self._refresh_executor = ThreadPoolExecutor(
            max_workers=max_refresh_workers,
            thread_name_prefix="timed-cache-refresh",
        )
        # Single lock guards _entries and all entry.is_refreshing flags.
        # It is never held across a fetch_fn call, so contention stays low.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, *args: Any, **kwargs: Any) -> T:
        """Return the cached value for the given arguments.

        Blocks only on a cold start (no entry yet).  Returns a stale value
        immediately if a background refresh is already running or has just
        been kicked off.
        """
        # Build key once so every branch (lookup, fetch, refresh) refers to the
        # exact same identity for this call.
        key = self._key_fn(*args, **kwargs)

        with self._lock:
            try:
                entry = self._entries.get(key)
            except TypeError as error:
                self._raise_unhashable_key_error(error)

            if entry is None:
                # Cold start — insert a placeholder and become the fetch owner.
                # The placeholder immediately serializes concurrent callers for
                # this key so we avoid duplicate upstream fetches.
                entry = _CacheEntry(value=None, fetched_at=None)
                self._entries[key] = entry
                self._entries.move_to_end(key, last=True)
                self._evict_one_if_needed_locked()
                owner = True
            else:
                owner = False
                # Any read marks the key as most recently used.
                self._entries.move_to_end(key, last=True)
                if self._is_stale(entry) and not entry.is_refreshing:
                    # Stale — return the old value now, refresh in background.
                    # Marking is_refreshing under the lock guarantees only one
                    # thread schedules refresh work for this key.
                    entry.is_refreshing = True
                    self._spawn_background_refresh(
                        key,
                        entry,
                        args,
                        kwargs,
                    )

        if owner:
            # Owner thread performs the initial fetch outside the lock.
            return self._do_cold_fetch(key, entry, args, kwargs)

        # For warm or stale-but-already-refreshing entries this is a no-op.
        # For concurrent cold-start threads it blocks until the owner is done.
        entry.ready.wait()
        with self._lock:
            if entry.error is not None:
                # cold-start failures are propagated to all waiters
                raise entry.error
            return entry.value

    def peek(self, *args: Any, **kwargs: Any) -> T | None:
        """Return a cached value if present, without triggering a fetch.

        Returns ``None`` when the key is absent or still in an in-flight cold
        fetch placeholder state.
        """
        key = self._key_fn(*args, **kwargs)
        with self._lock:
            try:
                entry = self._entries.get(key)
            except TypeError as error:
                self._raise_unhashable_key_error(error)
            if entry is None or not entry.ready.is_set():
                return None
            return entry.value

    def invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Remove the cache entry for the given arguments, if present."""
        key = self._key_fn(*args, **kwargs)
        with self._lock:
            try:
                self._entries.pop(key, None)
            except TypeError as error:
                self._raise_unhashable_key_error(error)

    def invalidate_all(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._entries.clear()

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        """Manually trigger a background refresh for the given arguments.

        If the entry is already being refreshed, this is a no-op.
        If an initial cold fetch is in-flight, this is also a no-op.
        If the entry does not exist, it will be fetched in the background.
        """
        key = self._key_fn(*args, **kwargs)
        with self._lock:
            try:
                entry = self._entries.get(key)
            except TypeError as error:
                self._raise_unhashable_key_error(error)
            if entry is None:
                # Cold start in background
                # Callers of refresh() explicitly asked for non-blocking behavior,
                # so we create the placeholder and let the pool populate it.
                entry = _CacheEntry(value=None, fetched_at=None)
                self._entries[key] = entry
                self._entries.move_to_end(key, last=True)
                self._evict_one_if_needed_locked()
                entry.is_refreshing = True
                self._spawn_background_refresh(key, entry, args, kwargs)
            elif entry.ready.is_set() and not entry.is_refreshing:
                entry.is_refreshing = True
                self._spawn_background_refresh(key, entry, args, kwargs)

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_unhashable_key_error(error: TypeError) -> None:
        """Raise a helpful error for unhashable cache keys."""
        raise TypeError(
            f"Cache key is unhashable: {error}. "
            "The default 'make_key' requires hashable arguments (int, str, tuple). "
            "To support mutable arguments (list, dict, set), use 'key_fn=TimedCache.deep_key_fn'.",
        ) from None

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> Any:
        """A fast, non-recursive key generator (default).

        This assumes all arguments are already hashable and does not perform
        recursive normalization. Use this when performance is critical and
        arguments are simple (e.g., strings, ints, or already hashable tuples).
        """
        if not kwargs:
            return args
        return (args, tuple(sorted(kwargs.items())))

    @staticmethod
    def deep_key_fn(*args: Any, **kwargs: Any) -> Any:
        """A robust key generator that handles mutable arguments.

        Recursively converts lists, dicts, and sets into hashable equivalents
        (tuples) so they can be used as cache keys. Use this if your cached
        function accepts mutable arguments.
        """

        def make_hashable(obj: Any) -> Any:
            """Recursively convert mutable containers to hashable equivalents."""
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            if isinstance(obj, list):
                return tuple(make_hashable(i) for i in obj)
            if isinstance(obj, set):
                normalized = [make_hashable(i) for i in obj]
                try:
                    return tuple(sorted(normalized))
                except TypeError:
                    # Mixed types may not be directly comparable; sort by repr
                    # to keep ordering deterministic for equivalent sets.
                    return tuple(sorted(normalized, key=repr))
            return obj

        hashable_args = tuple(make_hashable(a) for a in args)
        hashable_kwargs = tuple(
            sorted((k, make_hashable(v)) for k, v in kwargs.items()),
        )
        return (hashable_args, hashable_kwargs)

    def _is_stale(self, entry: _CacheEntry[T]) -> bool:
        """Return True when a fetched entry has exceeded its TTL."""
        # Caller must hold self._lock.
        if entry.fetched_at is None:
            # Placeholder during first fetch: callers must wait on ready, not refresh.
            return False
        return (time.monotonic() - entry.fetched_at) >= self._ttl_seconds

    def _do_cold_fetch(
        self,
        key: Any,
        entry: _CacheEntry[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        """Perform the first fetch for a key, updating the entry in place.

        On failure the placeholder entry is removed so the next caller can try
        again, and the exception propagates to the caller.
        """
        try:
            value: T = self._fetch_fn(*args, **kwargs)
        except Exception as error:
            with self._lock:
                entry.error = error
                self._entries.pop(key, None)
            entry.ready.set()  # unblock any waiting threads so they can retry
            raise

        with self._lock:
            entry.error = None
            entry.value = value
            # monotonic avoids issues with wall-clock adjustments
            entry.fetched_at = time.monotonic()
        entry.ready.set()
        return value

    def _spawn_background_refresh(
        self,
        key: Any,
        entry: _CacheEntry[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Submit one stale-entry refresh into the bounded executor."""
        # Caller must hold self._lock and have set entry.is_refreshing = True.
        try:
            self._refresh_executor.submit(
                self._background_refresh,
                key,
                entry,
                args,
                kwargs,
            )
        except Exception:
            # Never leave the entry stuck in refreshing state.
            # If submit fails (e.g., executor shutdown), callers must be able
            # to trigger refresh again later.
            entry.is_refreshing = False
            raise

    def _background_refresh(
        self,
        key: Any,
        entry: _CacheEntry[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Refresh a stale entry and keep old data if refresh fails."""
        try:
            new_value: T = self._fetch_fn(*args, **kwargs)
            with self._lock:
                entry.is_refreshing = False
                entry.value = new_value
                entry.fetched_at = time.monotonic()
                entry.error = None
        except Exception as error:
            fn_name = getattr(self._fetch_fn, "__qualname__", repr(self._fetch_fn))
            logger.exception(
                "Background refresh failed for %s; keeping stale value",
                fn_name,
            )
            with self._lock:
                entry.is_refreshing = False  # must clear even on failure
                # Cold-key background refresh failed: remove placeholder so
                # next get() retries instead of returning None.
                if not entry.ready.is_set():
                    entry.error = error
                    if self._entries.get(key) is entry:
                        # Identity check avoids deleting a newer replacement
                        # entry that may have been inserted concurrently.
                        self._entries.pop(key, None)
        finally:
            entry.ready.set()

    def _evict_one_if_needed_locked(self) -> None:
        """Evict one entry if configured max size is exceeded.

        Caller must hold self._lock.
        """
        if self._max_entries is None or len(self._entries) <= self._max_entries:
            return

        self._entries.popitem(last=False)


class TimedCollection(TimedCache[V], Generic[K, V]):
    """A TimedCache variant that behaves like a dictionary for a set of keys.

    Each key is fetched and cached independently, allowing individual TTLs.
    The fetch function must accept an iterable of keys and optional kwargs:
    ``fetch_fn(keys, **kwargs)``.

    Key Generation:
    By default, this uses the fast ``make_key`` which expects keys to be
    hashable. Pass ``key_fn=TimedCache.deep_key_fn`` if keys are mutable.
    """

    def __init__(
        self,
        fetch_fn: Callable[..., dict[K, V]],
        ttl_seconds: float = 300,
        max_entries: int | None = None,
        max_refresh_workers: int = 8,
        key_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._batch_fetch_fn = fetch_fn
        # Adapter for TimedCache's single-key get/refresh operations.
        super().__init__(
            fetch_fn=self._fetch_single,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            max_refresh_workers=max_refresh_workers,
            key_fn=key_fn or self._make_collection_key,
        )

    @staticmethod
    def _make_collection_key(key: Any, **kwargs: Any) -> Any:
        """Default key generator for TimedCollection.

        Wraps the key in a tuple to match TimedCache's expectation of *args.
        """
        return TimedCache.make_key(key, **kwargs)

    def _fetch_single(self, *args: Any, **kwargs: Any) -> V:
        """Adapter to call the batch fetch function for a single key."""
        # TimedCache always calls fetch_fn with the same *args passed to get().
        # TimedCollection expects single-key entries, but we must handle *args
        # to avoid TypeErrors from the underlying TimedCache.get() call.
        key = args[0] if len(args) == 1 else args
        results = self._batch_fetch_fn([key], **kwargs)
        if key not in results:
            raise KeyError(f"Key {key!r} missing from fetch_fn results")
        return results[key]

    def get_collection(self, keys: Iterable[K], **kwargs: Any) -> dict[K, V]:
        """Update active keys and return a dictionary of their current values.

        Efficiency:
        - Cold keys (not in cache) are fetched synchronously in a single batch.
        - Stale keys are returned immediately and refreshed in the background
          in a single batch.
        - Fresh keys are returned immediately.
        """
        # Materialize once so we preserve input order in the final response and
        # can iterate multiple times without exhausting generators.
        keys_list = list(keys)

        cold_keys: list[K] = []
        cold_keys_set: set[K] = set()
        # Tracks the exact placeholder object created by this call for each cold key.
        # If LRU eviction replaces/removes the dict entry mid-fetch, we still need to
        # complete that original object to unblock threads already waiting on it.
        owned_cold_entries: dict[K, _CacheEntry[V]] = {}
        stale_keys: list[K] = []
        results: dict[K, V] = {}
        waiting_entries: dict[K, _CacheEntry[V]] = {}

        with self._lock:
            for k in keys_list:
                ckey = self._key_fn(k, **kwargs)
                try:
                    entry = self._entries.get(ckey)
                except TypeError as error:
                    self._raise_unhashable_key_error(error)
                if entry is None:
                    # Claim ownership for this cold key so concurrent callers
                    # wait on this placeholder instead of re-fetching.
                    entry = _CacheEntry(value=None, fetched_at=None)
                    self._entries[ckey] = entry
                    self._entries.move_to_end(ckey, last=True)
                    self._evict_one_if_needed_locked()
                    owned_cold_entries[k] = entry
                    if k not in cold_keys_set:
                        cold_keys.append(k)
                        cold_keys_set.add(k)
                elif not entry.ready.is_set():
                    self._entries.move_to_end(ckey, last=True)
                    # Another fetch is in progress for this key.
                    waiting_entries.setdefault(k, entry)
                elif self._is_stale(entry):
                    # stale-while-revalidate: return stale now, refresh later
                    results[k] = entry.value
                    if not entry.is_refreshing:
                        stale_keys.append(k)
                        entry.is_refreshing = True
                    self._entries.move_to_end(ckey, last=True)
                else:
                    results[k] = entry.value
                    self._entries.move_to_end(ckey, last=True)

        if cold_keys:
            # Synchronous batch fetch for cold keys we own.
            try:
                new_data = self._batch_fetch_fn(cold_keys, **kwargs)
            except Exception as error:
                with self._lock:
                    for k in cold_keys:
                        owned_entry = owned_cold_entries[k]
                        ckey = self._key_fn(k, **kwargs)
                        try:
                            entry = self._entries.get(ckey)
                        except TypeError as key_error:
                            self._raise_unhashable_key_error(key_error)
                        if entry is not None and not entry.ready.is_set():
                            # The currently indexed placeholder still belongs
                            # to an in-flight cold fetch path.
                            entry.error = error
                            self._entries.pop(ckey, None)
                            entry.ready.set()
                        elif not owned_entry.ready.is_set():
                            # The entry we created was already evicted/replaced; complete
                            # the original placeholder object so existing waiters wake up.
                            owned_entry.error = error
                            owned_entry.ready.set()
                raise

            missing_keys = [k for k in cold_keys if k not in new_data]
            missing_error = None
            if missing_keys:
                missing_error = KeyError(
                    f"Keys missing from fetch_fn results: {missing_keys!r}",
                )

            with self._lock:
                for k in cold_keys:
                    owned_entry = owned_cold_entries[k]
                    ckey = self._key_fn(k, **kwargs)
                    try:
                        entry = self._entries.get(ckey)
                    except TypeError as key_error:
                        self._raise_unhashable_key_error(key_error)
                    if k in new_data:
                        v = new_data[k]
                        if entry is None:
                            # Placeholder was evicted while fetch ran; recreate
                            # a ready entry so future reads can proceed.
                            entry = _CacheEntry(value=v, fetched_at=time.monotonic())
                            entry.ready.set()
                            self._entries[ckey] = entry
                            self._evict_one_if_needed_locked()
                        else:
                            entry.value = v
                            entry.fetched_at = time.monotonic()
                            entry.error = None
                            entry.ready.set()
                            entry.is_refreshing = False
                        if owned_entry is not entry and not owned_entry.ready.is_set():
                            # If waiters hold a stale placeholder reference, mirror the
                            # successful result onto it so their wait/read path still works.
                            owned_entry.value = v
                            owned_entry.fetched_at = time.monotonic()
                            owned_entry.error = None
                            owned_entry.is_refreshing = False
                            owned_entry.ready.set()
                        results[k] = v
                    elif entry is not None and not entry.ready.is_set():
                        entry.error = cast("Exception", missing_error)
                        self._entries.pop(ckey, None)
                        entry.ready.set()
                    elif not owned_entry.ready.is_set():
                        # Missing-key failure with evicted/replaced placeholder: propagate
                        # the same error to the original object to avoid indefinite waits.
                        owned_entry.error = cast("Exception", missing_error)
                        owned_entry.ready.set()

            if missing_error is not None:
                raise missing_error

        for k, entry in waiting_entries.items():
            # Wait outside lock to avoid blocking unrelated keys.
            entry.ready.wait()
            with self._lock:
                if entry.error is not None:
                    raise entry.error
                results[k] = entry.value

        if stale_keys:
            # Background batch refresh for stale keys
            try:
                self._refresh_executor.submit(self._batch_refresh, stale_keys, kwargs)
            except Exception:
                with self._lock:
                    for k in stale_keys:
                        ckey = self._key_fn(k, **kwargs)
                        try:
                            entry = self._entries.get(ckey)
                        except TypeError as key_error:
                            self._raise_unhashable_key_error(key_error)
                        if entry is not None:
                            entry.is_refreshing = False
                raise

        return {k: results[k] for k in keys_list if k in results}

    def peek_collection(self, keys: Iterable[K], **kwargs: Any) -> dict[K, V]:
        """Return cached values for the given keys if present, without triggering a fetch.

        Keys that are absent or still in an in-flight cold fetch state are omitted.
        """
        results: dict[K, V] = {}
        with self._lock:
            for k in keys:
                ckey = self._key_fn(k, **kwargs)
                try:
                    entry = self._entries.get(ckey)
                except TypeError as error:
                    self._raise_unhashable_key_error(error)
                if entry is not None and entry.ready.is_set():
                    results[k] = cast("V", entry.value)
        return results

    def invalidate_collection(self, keys: Iterable[K], **kwargs: Any) -> None:
        """Remove cache entries for the given keys and arguments, if present."""
        with self._lock:
            for k in keys:
                ckey = self._key_fn(k, **kwargs)
                try:
                    self._entries.pop(ckey, None)
                except TypeError as error:
                    self._raise_unhashable_key_error(error)

    def _batch_refresh(self, keys: list[K], kwargs: dict[str, Any]) -> None:
        """Background worker to refresh a batch of keys."""
        try:
            new_data = self._batch_fetch_fn(keys, **kwargs)
            with self._lock:
                for k in keys:
                    ckey = self._key_fn(k, **kwargs)
                    try:
                        entry = self._entries.get(ckey)
                    except TypeError as key_error:
                        self._raise_unhashable_key_error(key_error)
                    if entry is None:
                        continue
                    if k in new_data:
                        v = new_data[k]
                        entry.value = v
                        entry.fetched_at = time.monotonic()
                        entry.error = None
                        entry.is_refreshing = False
                        entry.ready.set()
                        continue

                    # Missing key during background refresh:
                    # - for warm entries keep stale value and just clear flag
                    # - for placeholders fail fast and remove unusable entry
                    entry.is_refreshing = False
                    if not entry.ready.is_set():
                        missing_error = KeyError(
                            f"Key {k!r} missing from fetch_fn results",
                        )
                        entry.error = missing_error
                        self._entries.pop(ckey, None)
                    entry.ready.set()
        except Exception as error:
            logger.exception("Background batch refresh failed")
            with self._lock:
                for k in keys:
                    ckey = self._key_fn(k, **kwargs)
                    try:
                        entry = self._entries.get(ckey)
                    except TypeError as key_error:
                        self._raise_unhashable_key_error(key_error)
                    if entry:
                        entry.is_refreshing = False
                        if not entry.ready.is_set():
                            entry.error = error
                            self._entries.pop(ckey, None)
                        entry.ready.set()


@overload
def timed_cache(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def timed_cache(
    *,
    ttl_seconds: float = 300,
    max_entries: int | None = None,
    max_refresh_workers: int = 8,
    key_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def timed_cache(
    func: Callable[P, R] | None = None,
    /,
    *,
    ttl_seconds: float = 300,
    max_entries: int | None = None,
    max_refresh_workers: int = 8,
    key_fn: Callable[..., Any] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function with TimedCache semantics.

    By default, this uses the fast ``make_key`` which assumes all arguments are
    hashable (e.g., int, str, tuple). If the decorated function must accept
    mutable arguments (e.g., list, dict, set), provide
    ``key_fn=TimedCache.deep_key_fn``.

    Can be used as ``@timed_cache`` or ``@timed_cache(ttl_seconds=...)``.
    The returned wrapped function also exposes:
    - ``cache``: underlying TimedCache instance
    - ``peek(*args, **kwargs)``
    - ``refresh(*args, **kwargs)``
    - ``invalidate(*args, **kwargs)``
    - ``invalidate_all()``
    """

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        # One cache instance per decorated function object.
        cache = TimedCache(
            fetch_fn=inner,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            max_refresh_workers=max_refresh_workers,
            key_fn=key_fn,
        )

        @wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cache.get(*args, **kwargs)

        # Expose cache operations on the wrapper for ergonomic control in
        # application code and tests.
        wrapper.cache = cache
        wrapper.peek = cache.peek
        wrapper.refresh = cache.refresh
        wrapper.invalidate = cache.invalidate
        wrapper.invalidate_all = cache.invalidate_all
        return cast("Callable[P, R]", wrapper)

    if func is not None:
        return decorate(func)
    return decorate
