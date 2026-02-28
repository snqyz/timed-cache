"""Thread-safe timed cache with stale-while-revalidate semantics."""

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, overload, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
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
    ready: threading.Event = field(default_factory=threading.Event)
    is_refreshing: bool = False


class TimedCache(Generic[T]):
    """Wraps a fetch function and caches its result per unique set of arguments.

    Each distinct combination of arguments gets its own cache entry with an
    independent TTL (default: 5 minutes).  When an entry becomes stale the
    *next* caller receives the stale value immediately and a background thread
    silently refreshes it; no caller ever blocks on a refresh.

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
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 when provided")
        if max_refresh_workers < 1:
            raise ValueError("max_refresh_workers must be >= 1")

        self._fetch_fn = fetch_fn
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[CacheKey, _CacheEntry[T]] = OrderedDict()
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
        key = self._make_key(args, kwargs)

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                # Cold start — insert a placeholder and become the fetch owner.
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
                    entry.is_refreshing = True
                    self._spawn_background_refresh(entry, args, kwargs)

        if owner:
            return self._do_cold_fetch(key, entry, args, kwargs)

        # For warm or stale-but-already-refreshing entries this is a no-op.
        # For concurrent cold-start threads it blocks until the owner is done.
        entry.ready.wait()
        with self._lock:
            return entry.value

    def invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Remove the cache entry for the given arguments, if present."""
        key = self._make_key(args, kwargs)
        with self._lock:
            self._entries.pop(key, None)

    def invalidate_all(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> CacheKey:
        """Normalize call arguments into a deterministic, hashable cache key."""

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
        key: CacheKey,
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
        except Exception:
            with self._lock:
                self._entries.pop(key, None)
            entry.ready.set()  # unblock any waiting threads so they can retry
            raise

        with self._lock:
            entry.value = value
            entry.fetched_at = time.monotonic()
        entry.ready.set()
        return value

    def _spawn_background_refresh(
        self,
        entry: _CacheEntry[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Submit one stale-entry refresh into the bounded executor."""
        # Caller must hold self._lock and have set entry.is_refreshing = True.
        try:
            self._refresh_executor.submit(self._background_refresh, entry, args, kwargs)
        except Exception:
            # Never leave the entry stuck in refreshing state.
            entry.is_refreshing = False
            raise

    def _background_refresh(
        self,
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
        except Exception:
            fn_name = getattr(self._fetch_fn, "__qualname__", repr(self._fetch_fn))
            logger.exception(
                "Background refresh failed for %s; keeping stale value",
                fn_name,
            )
            with self._lock:
                entry.is_refreshing = False  # must clear even on failure

    def _evict_one_if_needed_locked(self) -> None:
        """Evict one entry if configured max size is exceeded.

        Caller must hold self._lock.
        """
        if self._max_entries is None or len(self._entries) <= self._max_entries:
            return

        self._entries.popitem(last=False)


@overload
def timed_cache(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def timed_cache(
    *,
    ttl_seconds: float = 300,
    max_entries: int | None = None,
    max_refresh_workers: int = 8,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def timed_cache(
    func: Callable[P, R] | None = None,
    /,
    *,
    ttl_seconds: float = 300,
    max_entries: int | None = None,
    max_refresh_workers: int = 8,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function with TimedCache semantics.

    Can be used as ``@timed_cache`` or ``@timed_cache(ttl_seconds=...)``.
    The returned wrapped function also exposes:
    - ``cache``: underlying TimedCache instance
    - ``invalidate(*args, **kwargs)``
    - ``invalidate_all()``
    """

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        cache = TimedCache(
            fetch_fn=inner,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            max_refresh_workers=max_refresh_workers,
        )

        @wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cache.get(*args, **kwargs)

        setattr(wrapper, "cache", cache)
        setattr(wrapper, "invalidate", cache.invalidate)
        setattr(wrapper, "invalidate_all", cache.invalidate_all)
        return cast(Callable[P, R], wrapper)

    if func is not None:
        return decorate(func)
    return decorate
