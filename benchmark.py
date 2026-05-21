import asyncio
import httpx
import time

async def send_request(client, url, payload):
    start_time = time.time()

    response = await client.post(url, json=payload)
    
    end_time = time.time()

    latency = end_time - start_time

    return latency

async def benchmark(url, payload, N):
    async with httpx.AsyncClient(timeout=30.0) as client:
        coroutines = [send_request(client, url, payload) for _ in range(N)]
        latencies = await asyncio.gather(*coroutines)
    return latencies

def compute_stats(latencies):
    latencies.sort()
    N = len(latencies)

    p50_index = int(0.5 * N)
    p50_latency = latencies[p50_index]

    p95_index = int(0.95 * N)
    p95_latency = latencies[p95_index]

    p99_index = int(0.99 * N)
    p99_latency = latencies[p99_index]

    return (p50_latency, p95_latency, p99_latency)

async def main():
    url = "http://127.0.0.1:8000/prediction"
    payload = {"reviews": ["The product was faulty"]}
    N = 10

    start_time = time.time()
    latencies = await benchmark(url, payload, N)
    end_time = time.time()

    p50_latency, p95_latency, p99_latency = compute_stats(latencies)

    total_time = end_time - start_time

    throughput = N / total_time

    print(f"p50 latency: {p50_latency:.2f}s")
    print(f"p95 latency: {p95_latency:.2f}")
    print(f"p99 latency: {p99_latency:.2f}s")
    print(f"throughput: {throughput:.2f} requests/second")


if __name__ == "__main__":
    asyncio.run(main())







