import asyncio
import aiohttp
import time
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

PROMPTS = [
	"What is machine learning?",
	"Explain neural networks simply",
	"What is the transformer architecture?",
	"Explain KV cache in LLMs",
	"What is retrieval augmented generation?",
	"Explain mixture of experts",
	"What is quantization in LLMs?",
	"Explain continuous batching",
	"What is PagedAttention?",
	"What is model serving?"
]

async def send_request(session, prompt, request_id):
	start_time = time.time()
	payload = {"prompt": prompt, "max_tokens": 100}

	try:
		async with session.post(
			f"{BASE_URL}/generate",
			json=payload
		) as response:
			result = await response.json()
			end_time = time.time()
			latency = end_time - start_time
			print(f"Request {request_id} completed in {latency:.3f}s")
			return latency
	
	except Exception as e:
		print(f"Request {request_id} failed: {e}")
		return None

async def load_test(concurrent_requests: int):
	print(f"Sending {concurrent_requests} concurrent requests...")
	
	async with aiohttp.ClientSession() as session:
		tasks = [
			send_request(session, PROMPTS[i % len(PROMPTS)], i)
			for i in range(concurrent_requests)
		]
		latencies = await asyncio.gather(*tasks)
	
	latencies = [l for l in latencies if l is not None]
	
	print(f"Results for {concurrent_requests} concurrent requests:")
	print(f"  Total requests:      {len(latencies)}")
	print(f"  P50 latency:         {np.percentile(latencies, 50):.3f}s")
	print(f"  P95 latency:         {np.percentile(latencies, 95):.3f}s")
	print(f"  P99 latency:         {np.percentile(latencies, 99):.3f}s")
	print(f"  Avg latency:         {np.mean(latencies):.3f}s")
	print(f"  Max latency:         {np.max(latencies):.3f}s")

async def main():
	for concurrency in [1, 3, 5]:
		await load_test(concurrency)
		print("-" * 40)

if __name__ == "__main__":
	asyncio.run(main())
