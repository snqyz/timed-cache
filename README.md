# timed-cache

`timed-cache` is a thread-safe in-memory cache with:

- TTL-based expiration per key
- stale-while-revalidate background refresh
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

For a runnable demo, see `examples/basic_usage.py`.
