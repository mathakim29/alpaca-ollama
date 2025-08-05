import httpx
import json
import numpy as np
from typing import List, Dict, Any

EMBED_MODEL = "snowflake-arctic-embed2:latest"
PROMPT_MODEL = "granite3.3:2b"

TIMEOUT = httpx.Timeout(70.0)

# embed generation
def embed_gen(query: str):
    payload = {
        "model": EMBED_MODEL,
        "input": query
    }
    try:
        with httpx.stream(
            "POST",
            "http://localhost:11434/api/embed",
            json=payload,
            timeout=TIMEOUT
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    return data.get("embeddings")
    except httpx.ReadTimeout:
        print("❌ Request timed out.")
    except httpx.RequestError as e:
        print(f"❌ Request failed: {e}")

def sim_check(query: str, sentences: List[str]) -> Dict[str, Any]:
    if not query or not sentences:
        return {
            "error": "Query and at least one comparison sentence are required."
        }

    query_embed = embed_gen(query)
    if not query_embed:
        return {
            "error": "Failed to embed the query sentence."
        }

    results = []
    max_similarity = -float("inf")
    best_match = None

    for i, sentence in enumerate(sentences):
        embed = embed_gen(sentence)
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

def prompt_gen(query: str):
    # Define the Granite chat payload
    payload = {
        "model": "granite3.3:2b",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        'stream': False,
        'raw': True
    }

    # Send a streaming chat request to Granite
    try:
        with httpx.stream(
            "POST",
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=TIMEOUT
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    return data
    except httpx.ReadTimeout:
        print("❌ Request timed out (read timeout after 70 seconds).")
    except httpx.RequestError as e:
        print(f"❌ Request failed: {e}")
