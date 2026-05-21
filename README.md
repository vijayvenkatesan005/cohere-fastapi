# Cohere FastAPI Sentiment Analysis

A production-ready sentiment analysis API built with FastAPI and Cohere's LLM API.

## Features
- FastAPI REST endpoint for sentiment classification
- Async concurrent request handling with asyncio
- Dockerized for portable deployment
- Prometheus metrics exposed for monitoring
- Async benchmarking script for measuring p50/p95/p99 latency and throughput

## Stack
- FastAPI + uvicorn
- Cohere API (Command A)
- Docker + Docker Compose
- Prometheus + Grafana
- httpx + asyncio

## Running locally
```bash
docker compose up
```

## Benchmarking
```bash
python benchmark.py
```
