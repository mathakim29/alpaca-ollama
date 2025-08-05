import httpx
import json
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Callable
from contextlib import asynccontextmanager
from functools import wraps
from fastapi import FastAPI, Query
from pydantic import BaseModel
from .constants import TIMEOUT, EMBED_MODEL, PROMPT_MODEL

# --------------------------------------
# Setup Logging
# --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------
# Constants for Model Configuration
# --------------------------------------
EMBED_MODEL = "snowflake-arctic-embed2:latest"
PROMPT_MODEL = "granite3.3:2b"
TIMEOUT = 70  # seconds

# --------------------------------------
# Async Decorator
# --------------------------------------
def async_fn(fn: Callable) -> Callable:
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        logger.debug(f"Calling async function: {fn.__name__}")
        return await fn(*args, **kwargs)
    return wrapper

# --------------------------------------
# Async Context Manager for HTTP Streaming
# --------------------------------------
@asynccontextmanager
async def async_http_stream(method: str, url: str, json_payload: dict, timeout: int):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(method, url, json=json_payload) as response:
                yield response
    except httpx.ReadTimeout:
        logger.error("Async request timed out.")
        yield None
    except httpx.RequestError as e:
        logger.error(f"Async request failed: {e}")
        yield None

# --------------------------------------
# Async Embedding Generator
# --------------------------------------
@async_fn
async def embed_gen(query: str) -> List[float] | None:
    payload = {"model": EMBED_MODEL, "input": query}
    async with async_http_stream("POST", "http://localhost:11434/api/embed", payload, TIMEOUT) as response:
        if response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    return data.get("embeddings")
    return None

# --------------------------------------
# Async Similarity Checker
# --------------------------------------
@async_fn
async def sim_check(query: str, sentences: List[str]) -> Dict[str, Any]:
    if not query or not sentences:
        return {"error": "Query and at least one comparison sentence are required."}

    query_embed = await embed_gen(query)
    if not query_embed:
        return {"error": "Failed to embed the query sentence."}

    sentence_embeds = await asyncio.gather(
        *(embed_gen(sentence) for sentence in sentences)
    )

    results = []
    max_similarity = -float("inf")
    best_match = None

    for i, (sentence, embed) in enumerate(zip(sentences, sentence_embeds)):
        if not embed:
            similarity = None
        else:
            similarity = float(np.dot(np.array(query_embed).flatten(), np.array(embed).flatten()))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = {
                    "index": i,
                    "sentence": sentence,
                    "similarity": similarity
                }
        results.append({"index": i, "sentence": sentence, "similarity": similarity})

    return {"query": query, "comparisons": results, "best": best_match}

# --------------------------------------
# Async Prompt Generator
# --------------------------------------
@async_fn
async def prompt_gen(query: str, stream: bool = False, JSON_OUTPUT: bool = True) -> Dict[str, Any] | None:
    payload = {
        "model": PROMPT_MODEL,
        "messages": [{"role": "user", "content": query}],
        'stream': stream,
        'raw': True,
        'json': JSON_OUTPUT
    }
    async with async_http_stream("POST", "http://localhost:11434/api/chat", payload, TIMEOUT) as response:
        if response:
            if stream:
                full_content = ""
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        delta = data.get("message", {}).get("content", "")
                        full_content += delta
                return {"content": full_content}
            else:
                async for line in response.aiter_lines():
                    if line:
                        return json.loads(line)
    return None


