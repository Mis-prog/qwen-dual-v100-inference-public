"""Measure end-to-end RPS, latency, output tokens/s for two llama-server replicas."""
import argparse
import asyncio
import math
import time

from infer import DualGPU


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


async def benchmark(total: int, concurrency: int, max_tokens: int) -> None:
    if total < 1 or concurrency < 1 or max_tokens < 1:
        raise ValueError("total, concurrency and max-tokens must be positive")
    gate = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    total_tokens = 0
    errors = 0

    async with DualGPU() as llm:
        # Warm up both GPU replicas before the timed section.
        await asyncio.gather(
            llm.generate("Привет", max_tokens=8),
            llm.generate("Привет", max_tokens=8),
        )

        async def worker(i: int) -> None:
            nonlocal total_tokens, errors
            async with gate:
                started = time.perf_counter()
                try:
                    result = await llm.generate(
                        f"Объясни принцип работы Transformer кратко. Запрос №{i}",
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    errors += 1
                    print(f"Request {i} failed: {exc}")
                    return
                latencies.append(time.perf_counter() - started)
                total_tokens += result.get("usage", {}).get("completion_tokens", 0)

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total)))
        elapsed = time.perf_counter() - started

    print(f"Concurrency: {concurrency}")
    print(f"Requests: {total} | success: {len(latencies)} | errors: {errors}")
    print(f"Elapsed: {elapsed:.2f}s | RPS: {len(latencies) / elapsed:.2f}")
    print(f"Output tokens/s: {total_tokens / elapsed:.2f}")
    if latencies:
        print(f"p50 latency: {percentile(latencies, 0.50):.2f}s")
        print(f"p95 latency: {percentile(latencies, 0.95):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    asyncio.run(benchmark(args.total, args.concurrency, args.max_tokens))
