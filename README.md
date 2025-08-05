Here's the cleaned-up version of your `README.md` with **no emojis** and includes the ability to set default models:
# Alpaca - Async Ollama API Library with Extras

## Features

- Fully async using `httpx` and `asyncio`
- Embedding generation with similarity ranking
- Prompt generation with optional streaming
- Customizable default model configuration

## Installation

1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
````

2. Clone this repository

3. Place the `alpaca/` folder in the same directory as your Python script

## Default Models

```python
EMBED_MODEL = "snowflake-arctic-embed2:latest"
PROMPT_MODEL = "granite3.3:2b"
```

You can override the default models at runtime:

```python
from alpaca import api

api.EMBED_MODEL = "your-custom-embed-model"
api.PROMPT_MODEL = "your-custom-prompt-model"
```

## Usage

### Prompt Generation

```python
import asyncio
from alpaca import api

async def main():
    prompt = "Explain the importance of clean code."
    result = await api.prompt_gen(prompt)
    print(result)

asyncio.run(main())
```

### Embedding Similarity

```python
import asyncio
from alpaca import api

async def main():
    query = "What is the capital of France?"
    candidates = [
        "Paris is the capital of France.",
        "Berlin is in Germany.",
        "Madrid is in Spain."
    ]
    result = await api.sim_check(query, candidates)
    print(result)

asyncio.run(main())
```

### Streaming Prompts

```python
response = await api.prompt_gen("Tell me a story", stream=True)
print(response["content"])
```

## Notes
* For more detailed examples and output, see the [examples.ipynb](./examples.ipynb) notebook.
* Requires Ollama running at `http://localhost:11434`
* All HTTP calls are streamed using `httpx.AsyncClient`

## Dependencies

```bash
pip install fastapi httpx numpy pydantic
```

## Contributing

Contributions are welcome. Please open issues or submit pull requests.

## License

MIT – see the [LICENSE](LICENSE) file for details.

