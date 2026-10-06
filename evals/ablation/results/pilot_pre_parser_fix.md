## Outcomes by model and surface

| Model | Surface | Correct | Confidently wrong | (silently wrong) | Safe failure |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | v04 | 8/72 = 11% [6%, 20%] | 38/72 = 53% [41%, 64%] | 1/72 = 1% [0%, 7%] | 26/72 = 36% [26%, 48%] |
| ollama:llama3.1:latest | v05 | 10/72 = 14% [8%, 24%] | 38/72 = 53% [41%, 64%] | 0/72 = 0% [0%, 5%] | 24/72 = 33% [24%, 45%] |
| ollama:qwen3.5:9b | v04 | 27/72 = 38% [27%, 49%] | 37/72 = 51% [40%, 63%] | 26/72 = 36% [26%, 48%] | 8/72 = 11% [6%, 20%] |
| ollama:qwen3.5:9b | v05 | 40/72 = 56% [44%, 66%] | 17/72 = 24% [15%, 35%] | 1/72 = 1% [0%, 7%] | 15/72 = 21% [13%, 32%] |

## Correct rate by category

| Model | Surface | control | dropped | flag | trade |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | v04 | 0/24 | 0/21 | 4/15 | 4/12 |
| ollama:llama3.1:latest | v05 | 1/24 | 0/21 | 5/15 | 4/12 |
| ollama:qwen3.5:9b | v04 | 22/24 | 0/21 | 4/15 | 1/12 |
| ollama:qwen3.5:9b | v05 | 21/24 | 13/21 | 4/15 | 2/12 |

## Paired comparison (v05 vs v04, same task and repeat)

| Model | Pairs | v05 only correct | v04 only correct | McNemar p |
|---|---|---|---|---|
| ollama:llama3.1:latest | 72 | 7 | 5 | 0.774 |
| ollama:qwen3.5:9b | 72 | 19 | 6 | 0.0146 |

## Cost drivers per run

| Model | Surface | Tool calls | Failed calls | LLM calls | Prompt tok | Completion tok |
|---|---|---|---|---|---|---|
| ollama:llama3.1:latest | v04 | 1.0 | 1.03 | 1.6 | 5,309 | 766 |
| ollama:llama3.1:latest | v05 | 1.0 | 0.93 | 1.6 | 5,409 | 692 |
| ollama:qwen3.5:9b | v04 | 1.4 | 0.19 | 2.4 | 16,348 | 1,659 |
| ollama:qwen3.5:9b | v05 | 2.8 | 1.25 | 3.6 | 24,960 | 1,898 |
