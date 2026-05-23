import asyncio
import time
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def calculate(expression):
	return eval(expression)

def send_tool_request(payload):
    messages = [{"role": "user", "content": payload}]
    
    # Stage 1 - time to first tool call
    start_time = time.time()
    response = co.chat(model="command-a-03-2025", messages=messages, tools=tools)
    end_time = time.time()
    first_tool_call_time = end_time - start_time

    # Stage 2 - tool execution time
    start_time = time.time()
    tool_call = response.message.tool_calls[0]
    tool_args = eval(tool_call.function.arguments)
    tool_result = calculate(tool_args["expression"])
    end_time = time.time()
    tool_execution_time = end_time - start_time

    # Stage 3 - time for final response
    start_time = time.time()
    messages.append({"role": "assistant", "tool_calls": response.message.tool_calls, "tool_plan": response.message.tool_plan})
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)})
    response = co.chat(model="command-a-03-2025", messages=messages, tools=tools)
    end_time = time.time()
    final_response_time = end_time - start_time

    return {
	"first_tool_call": first_tool_call_time,
	"tool_execution": tool_execution_time,
	"final_response": final_response_time,
	"total": first_tool_call_time + tool_execution_time + final_response_time
    }

async def benchmark_tool_use(payload, N):
	coroutines = [asyncio.to_thread(send_tool_request, payload) for _ in range(N)]
	results = await asyncio.gather(*coroutines)
	return results

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
    payload = "What is 1234 * 5678?"
    N = 10

    start_time = time.time()
    results = await benchmark_tool_use(payload, N)
    end_time = time.time()
    total_time = end_time - start_time

    first_tool_call_latencies = [r["first_tool_call"] for r in results]
    tool_execution_latencies = [r["tool_execution"] for r in results]
    final_response_latencies = [r["final_response"] for r in results]
    total_latencies = [r["total"] for r in results]

    p50_ftc, p95_ftc, p99_ftc = compute_stats(first_tool_call_latencies)
    p50_te, p95_te, p99_te = compute_stats(tool_execution_latencies)
    p50_fr, p95_fr, p99_fr = compute_stats(final_response_latencies)
    p50_tot, p95_tot, p99_tot = compute_stats(total_latencies)
    
    throughput = N / total_time
    
    print(f"\n{'Stage':<20} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 46)
    print(f"{'First tool call':<20} {p50_ftc:>8.2f}s {p95_ftc:>8.2f}s {p99_ftc:>8.2f}s")
    print(f"{'Tool execution':<20} {p50_te:>8.2f}s {p95_te:>8.2f}s {p99_te:>8.2f}s")
    print(f"{'Final response':<20} {p50_fr:>8.2f}s {p95_fr:>8.2f}s {p99_fr:>8.2f}s")
    print(f"{'Total':<20} {p50_tot:>8.2f}s {p95_tot:>8.2f}s {p99_tot:>8.2f}s")
    print(f"\nThroughput: {throughput:.2f} requests/second")
    
    

if __name__ == "__main__":
    asyncio.run(main())







