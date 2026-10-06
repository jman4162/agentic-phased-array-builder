## Outcomes by model and surface

| Model | Surface | Correct | Confidently wrong | (silently wrong) | Safe failure |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 8/72 = 11% [6%, 20%] | 36/72 = 50% [39%, 61%] | 0/72 = 0% [0%, 5%] | 28/72 = 39% [28%, 50%] |
| ollama:llama3.1:latest | v04 | 10/72 = 14% [8%, 24%] | 41/72 = 57% [45%, 68%] | 3/72 = 4% [1%, 12%] | 21/72 = 29% [20%, 41%] |
| ollama:llama3.1:latest | v05 | 2/72 = 3% [1%, 10%] | 43/72 = 60% [48%, 70%] | 3/72 = 4% [1%, 12%] | 27/72 = 38% [27%, 49%] |
| ollama:qwen3.5:9b | current | 50/72 = 69% [58%, 79%] | 15/72 = 21% [13%, 32%] | 0/72 = 0% [0%, 5%] | 7/72 = 10% [5%, 19%] |
| ollama:qwen3.5:9b | v04 | 30/72 = 42% [31%, 53%] | 33/72 = 46% [35%, 57%] | 23/72 = 32% [22%, 43%] | 9/72 = 12% [7%, 22%] |
| ollama:qwen3.5:9b | v05 | 40/72 = 56% [44%, 66%] | 19/72 = 26% [18%, 38%] | 0/72 = 0% [0%, 5%] | 13/72 = 18% [11%, 28%] |

## Correct rate by category

| Model | Surface | control | dropped | flag | trade |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 1/24 | 0/21 | 3/15 | 4/12 |
| ollama:llama3.1:latest | v04 | 3/24 | 0/21 | 6/15 | 1/12 |
| ollama:llama3.1:latest | v05 | 0/24 | 0/21 | 2/15 | 0/12 |
| ollama:qwen3.5:9b | current | 22/24 | 15/21 | 8/15 | 5/12 |
| ollama:qwen3.5:9b | v04 | 24/24 | 0/21 | 6/15 | 0/12 |
| ollama:qwen3.5:9b | v05 | 18/24 | 13/21 | 6/15 | 3/12 |

## Paired comparison (v05 vs v04, same task and repeat)

| Model | Pairs | v05 only correct | v04 only correct | McNemar p |
|---|---|---|---|---|
| ollama:llama3.1:latest | 72 | 1 | 9 | 0.0215 |
| ollama:qwen3.5:9b | 72 | 20 | 10 | 0.0987 |

## Cost drivers per run

| Model | Surface | Tool calls | Failed calls | LLM calls | Prompt tok | Completion tok |
|---|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 0.8 | 0.54 | 1.4 | 5,193 | 670 |
| ollama:llama3.1:latest | v04 | 1.1 | 0.40 | 1.5 | 5,580 | 646 |
| ollama:llama3.1:latest | v05 | 0.7 | 0.47 | 1.4 | 5,243 | 658 |
| ollama:qwen3.5:9b | current | 1.9 | 0.57 | 2.8 | 18,567 | 1,808 |
| ollama:qwen3.5:9b | v04 | 1.4 | 0.25 | 2.4 | 15,565 | 1,469 |
| ollama:qwen3.5:9b | v05 | 2.7 | 1.44 | 3.5 | 24,230 | 2,089 |
