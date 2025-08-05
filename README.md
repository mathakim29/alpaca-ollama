# Alpaca - Async Ollama API library with extras

## Features

- Async by default
- Uses modern Python async code.
- Utility functions like embedding similarity ranking.
- Generate prompts and get responses from models 

## Installation
1. Install ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
````

2. Clone the repository
3. Put alpaca folder in same directory as your script

## Default models:
EMBED_MODEL = "snowflake-arctic-embed2:latest"
PROMPT_MODEL = "granite3.3:2b"

## Usage

Prompt generating example

```python
import asyncio
from alpaca import api

async def main():
    prompt = "Explain the importance of clean code."
    response = await api.prompt_gen(prompt)
    print(response)

asyncio.run(main())
```

Embedding similarity checking example: 

```python
import asyncio
from alpaca import api

async def main():
    query = "What is the capital of France?"
    sentences = [
        "Paris is the capital of France.",
        "Berlin is in Germany.",
        "Madrid is in Spain."
    ]
    result = await api.sim_check(query, sentences)
    print(result)

asyncio.run(main())
```

For more detailed examples and output, see the [examples.ipynb](./examples.ipynb) notebook.

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.