from __future__ import annotations

from unittest.mock import MagicMock

from timed_cache import timed_cache


def test_timed_cache_decorator_without_parentheses_caches_values() -> None:
    mock = MagicMock(side_effect=lambda x: x * 10)

    @timed_cache
    def compute(x: int) -> int:
        return mock(x)

    assert compute(2) == 20
    assert compute(2) == 20
    assert mock.call_count == 1


def test_timed_cache_decorator_with_options_uses_same_key_normalization() -> None:
    mock = MagicMock(return_value=1)

    @timed_cache(ttl_seconds=60)
    def load(*, a: int, b: int) -> int:
        return mock(a=a, b=b)

    assert load(a=1, b=2) == 1
    assert load(b=2, a=1) == 1
    assert mock.call_count == 1


def test_decorated_function_exposes_invalidation_helpers() -> None:
    mock = MagicMock(side_effect=[10, 20])

    @timed_cache
    def fetch(value: int) -> int:
        return mock(value)

    assert fetch(1) == 10
    assert fetch(1) == 10

    fetch.invalidate(1)
    assert fetch(1) == 20

    assert mock.call_count == 2
    assert fetch.cache.size == 1
