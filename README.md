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
from timed_cache import timed_cache

@timed_cache(ttl_seconds=60)
def fetch_user(user_id: int) -> dict[str, int]:
    return {"id": user_id}

user = fetch_user(42)
```

Optional controls:

```python
from timed_cache import TimedCache

cache = TimedCache(
    fetch_fn=fetch_user,
    ttl_seconds=60,
    max_refresh_workers=4,   # bound concurrent stale refreshes
    max_entries=10_000,      # cap memory growth
)
```

Decorator helpers:

```python
fetch_user.peek(42)   # returns cached value or None; never fetches
fetch_user.invalidate(42)
fetch_user.invalidate_all()
size = fetch_user.cache.size
```

For a runnable demo, see `examples/basic_usage.py`.
