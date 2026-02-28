# timed-cache

`timed-cache` is a thread-safe in-memory cache with:

- TTL-based expiration per key
- stale-while-revalidate background refresh
- bounded background refresh worker pool
- optional max-size LRU eviction
- explicit invalidation APIs

## Install

```bash
uv sync
```

## Run tests

```bash
uv run pytest
```

## Quick usage

```python
from timed_cache import TimedCache

cache = TimedCache(fetch_fn=lambda user_id: {"id": user_id}, ttl_seconds=60)
user = cache.get(42)
```

Optional controls:

```python
cache = TimedCache(
    fetch_fn=fetch_user,
    ttl_seconds=60,
    max_refresh_workers=4,   # bound concurrent stale refreshes
    max_entries=10_000,      # cap memory growth
)
```

For a runnable demo, see `examples/basic_usage.py`.
