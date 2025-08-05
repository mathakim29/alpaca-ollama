import asyncio
import httpx
import json
import numpy as np
from typing import List, Dict, Any, Optional

EMBED_MODEL = "snowflake-arctic-embed2:latest"
PROMPT_MODEL = "granite3.3:2b"

TIMEOUT = httpx.Timeout(70.0)


# Async embed generation
async def embed_gen(query: str) -> Optional[List[float]]:
    payload = {
        "model": EMBED_MODEL,
        "input": query
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/embed",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        return data.get("embeddings")
    except httpx.ReadTimeout:
        print("❌ Request timed out.")
    except httpx.RequestError as e:
        print(f"❌ Request failed: {e}")
    return None


# Async similarity checker
async def sim_check(query: str, sentences: List[str]) -> Dict[str, Any]:
    if not query or not sentences:
        return {
            "error": "Query and at least one comparison sentence are required."
        }

    query_embed = await embed_gen(query)
    if not query_embed:
        return {
            "error": "Failed to embed the query sentence."
        }

    results = []
    max_similarity = -float("inf")
    best_match = None

    # Run embedding for each sentence sequentially (can be parallelized too)
    for i, sentence in enumerate(sentences):
        embed = await embed_gen(sentence)
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

        results.append({
            "index": i,
            "sentence": sentence,
            "similarity": similarity
        })

    return {
        "query": query,
        "comparisons": results,
        "best": best_match
    }


# Async prompt generation
async def prompt_gen(query: str) -> Optional[Dict[str, Any]]:
    payload = {
        "model": PROMPT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        'stream': False,
        'raw': True
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        return json.loads(line)
    except httpx.ReadTimeout:
        print("❌ Request timed out (read timeout after 70 seconds).")
    except httpx.RequestError as e:
        print(f"❌ Request failed: {e}")
    return None
