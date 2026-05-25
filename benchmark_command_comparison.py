import asyncio
import time
import cohere
import wandb
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def send_request(model, payload):
    messages = [{"role": "user", "content": payload}]
    
    start_time = time.time()
    response = co.chat(model=model, messages=messages)
    end_time = time.time()
    latency = end_time - start_time
    return latency

async def benchmark(model, payload, N):
    coroutines = [asyncio.to_thread(send_request, model, payload) for _ in range(N)]
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
    wandb.init(
    project="cohere-benchmark",
    name="command-a-vs-command-a-plus"
    )


    payload = "Explain what machine learning is in one sentence"
    N = 5
    models = ["command-a-03-2025", "command-a-plus-05-2026"]

    print(f"\n{'Model':<30} {'p50':>8} {'p95':>8} {'p99':>8} {'throughput':>12}")
    print("-" * 62)

    results = []

    
    for model in models:
        start_time = time.time()
        latencies = await benchmark(model, payload, N)
        end_time = time.time()
        total_time = end_time - start_time
        p50, p95, p99 = compute_stats(latencies)
        throughput = N / total_time
        print(f"{model:<30} {p50:>8.2f}s {p95:>8.2f}s {p99:>8.2f}s {throughput:>10.2f}/s")
        results.append([model, p50, p95, p99, throughput])
    
    table = wandb.Table(
    columns=["model", "p50", "p95", "p99", "throughput"],
    data=results
    )
    
    wandb.log({"benchmark_results": table})
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())







