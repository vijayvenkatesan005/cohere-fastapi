import cohere
import os
import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

load_dotenv()
co = cohere.AsyncClientV2(os.getenv("COHERE_API_KEY"))

app = FastAPI()

latency_store = []

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.post("/generate")
async def generate(request: GenerateRequest):
    start_time = time.time()
    first_token_time = None
    full_response = ""

    response = co.chat_stream(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=request.max_tokens
    )

    async for event in response:
        if event.type == "content-delta":
            if first_token_time is None:
                first_token_time = time.time()
            full_response += event.delta.message.content.text
    
    end_time = time.time()

    ttft = first_token_time - start_time
    total_latency = end_time - start_time
    latency_store.append(total_latency)

    return {
        "response": full_response,
        "ttft_seconds": round(ttft, 3),
        "total_latency_seconds": round(total_latency, 3)
    }

@app.get("/metrics")
async def metrics():
    if not latency_store:
        return {"message": "No requests yet"}

    return {
        "total_requests": len(latency_store),
        "p50_latency_seconds": round(float(np.percentile(latency_store, 50)), 3),
        "p95_latency_seconds": round(float(np.percentile(latency_store, 95)), 3),
        "p99_latency_seconds": round(float(np.percentile(latency_store, 99)), 3),
        "avg_latency_seconds": round(float(np.mean(latency_store)), 3),
        "min_latency_seconds": round(float(np.min(latency_store)), 3),
        "max_latency_seconds": round(float(np.max(latency_store)), 3),
    }
    