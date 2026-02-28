from collections.abc import Callable
from unittest.mock import MagicMock

from timed_cache import TimedCache


def test_returns_fetched_value(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    assert cache.get("a") == 1


def test_cache_hit_does_not_refetch(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get("a")
    cache.get("a")
    assert mock.call_count == 1


def test_different_args_fetch_independently(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    first = cache.get("a")
    second = cache.get("b")
    assert first != second
    assert mock.call_count == 2


def test_returns_typed_value() -> None:
    cache: TimedCache[str] = TimedCache(fetch_fn=lambda name: f"hello {name}")
    assert cache.get("world") == "hello world"


def test_same_positional_args_hit(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get(1, 2, 3)
    cache.get(1, 2, 3)
    assert mock.call_count == 1


def test_different_positional_args_miss(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get(1, 2)
    cache.get(1, 3)
    assert mock.call_count == 2


def test_kwargs_differentiate_keys(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get("x", flag=True)
    cache.get("x", flag=False)
    assert mock.call_count == 2


def test_kwarg_order_is_irrelevant(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get(a=1, b=2)
    cache.get(b=2, a=1)
    assert mock.call_count == 1


def test_dict_arg_is_hashable(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get({"key": "value"})
    cache.get({"key": "value"})
    assert mock.call_count == 1


def test_list_arg_is_hashable(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get([1, 2, 3])
    cache.get([1, 2, 3])
    assert mock.call_count == 1


def test_nested_dict_differentiates_keys(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get({"a": {"b": 1}})
    cache.get({"a": {"b": 2}})
    assert mock.call_count == 2


def test_no_args_is_valid_key(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    first = cache.get()
    second = cache.get()
    assert first == second
    assert mock.call_count == 1


def test_nested_kwargs_are_hashable_and_order_independent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get(config={"roles": ["admin", "user"], "flags": {"active": True}})
    cache.get(config={"flags": {"active": True}, "roles": ["admin", "user"]})
    assert mock.call_count == 1


def test_args_and_kwargs_produce_different_keys(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get(1, 2)
    cache.get(1, b=2)
    assert mock.call_count == 2


def test_invalidate_forces_refetch(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get("k")
    cache.invalidate("k")
    cache.get("k")
    assert mock.call_count == 2


def test_invalidate_nonexistent_key_is_silent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.invalidate("does_not_exist")


def test_invalidate_only_removes_matching_key(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get("a")
    cache.get("b")
    cache.invalidate("a")
    cache.get("a")
    cache.get("b")
    assert mock.call_count == 3


def test_invalidate_all_clears_everything(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    mock, cache = counter_cache_factory()
    cache.get("a")
    cache.get("b")
    cache.get("c")
    cache.invalidate_all()
    assert cache.size == 0
    cache.get("a")
    cache.get("b")
    assert mock.call_count == 5


def test_invalidate_all_on_empty_cache_is_silent(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.invalidate_all()


def test_empty_cache_has_size_zero(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    assert cache.size == 0


def test_size_increments_per_unique_key(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("a")
    assert cache.size == 1
    cache.get("b")
    assert cache.size == 2


def test_cache_hit_does_not_change_size(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("a")
    cache.get("a")
    assert cache.size == 1


def test_invalidate_decrements_size(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("a")
    cache.get("b")
    cache.invalidate("a")
    assert cache.size == 1


def test_invalidate_all_resets_size(
    counter_cache_factory: Callable[[float], tuple[MagicMock, TimedCache[int]]],
) -> None:
    _, cache = counter_cache_factory()
    cache.get("a")
    cache.get("b")
    cache.invalidate_all()
    assert cache.size == 0
