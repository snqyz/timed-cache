"""Small runnable example for TimedCache."""

from __future__ import annotations

import random
import threading
import time

from timed_cache import TimedCache


def fetch_scores(subject: str, *, passing_only: bool = False) -> list[int]:
    # Simulate an expensive upstream call.
    print(f"[fetch] fetching scores for {subject!r} (passing_only={passing_only})...")
    time.sleep(1)
    scores = [random.randint(50, 100) for _ in range(5)]
    return [score for score in scores if score >= 60] if passing_only else scores


def main() -> None:
    # Two requests for the same key ("math") should trigger only one cold fetch.
    cache: TimedCache[list[int]] = TimedCache(fetch_fn=fetch_scores, ttl_seconds=10)
    results: dict[str, list[int] | None] = {}

    def request(thread_name: str) -> None:
        # The second thread waits for the first cold fetch, then reuses result.
        value = cache.get("math")
        results[thread_name] = value
        print(f"[{thread_name}] got: {value}")

    t1 = threading.Thread(target=request, args=("request-1",))
    t2 = threading.Thread(target=request, args=("request-2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both threads should report the same cached payload.
    print("results:", results)


if __name__ == "__main__":
    main()
