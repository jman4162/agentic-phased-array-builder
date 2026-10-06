## Outcomes by model and surface

| Model | Surface | Correct | Confidently wrong | (silently wrong) | Safe failure |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 16/72 = 22% [14%, 33%] | 23/72 = 32% [22%, 43%] | 1/72 = 1% [0%, 7%] | 33/72 = 46% [35%, 57%] |
| ollama:llama3.1:latest | v04 | 13/72 = 18% [11%, 28%] | 35/72 = 49% [37%, 60%] | 5/72 = 7% [3%, 15%] | 24/72 = 33% [24%, 45%] |
| ollama:llama3.1:latest | v05 | 24/72 = 33% [24%, 45%] | 21/72 = 29% [20%, 41%] | 0/72 = 0% [0%, 5%] | 27/72 = 38% [27%, 49%] |
| ollama:qwen3.5:9b | current | 49/72 = 68% [57%, 78%] | 15/72 = 21% [13%, 32%] | 0/72 = 0% [0%, 5%] | 8/72 = 11% [6%, 20%] |
| ollama:qwen3.5:9b | v04 | 28/72 = 39% [28%, 50%] | 35/72 = 49% [37%, 60%] | 24/72 = 33% [24%, 45%] | 9/72 = 12% [7%, 22%] |
| ollama:qwen3.5:9b | v05 | 43/72 = 60% [48%, 70%] | 17/72 = 24% [15%, 35%] | 0/72 = 0% [0%, 5%] | 12/72 = 17% [10%, 27%] |

## Correct rate by category

| Model | Surface | control | dropped | flag | trade |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 5/24 | 0/21 | 9/15 | 2/12 |
| ollama:llama3.1:latest | v04 | 6/24 | 0/21 | 5/15 | 2/12 |
| ollama:llama3.1:latest | v05 | 8/24 | 3/21 | 11/15 | 2/12 |
| ollama:qwen3.5:9b | current | 23/24 | 15/21 | 6/15 | 5/12 |
| ollama:qwen3.5:9b | v04 | 22/24 | 0/21 | 6/15 | 0/12 |
| ollama:qwen3.5:9b | v05 | 19/24 | 12/21 | 8/15 | 4/12 |

## Paired comparisons (same task and repeat)

| Model | A vs B | Pairs | A only correct | B only correct | McNemar p |
|---|---|---|---|---|---|
| ollama:llama3.1:latest | v05 vs v04 | 72 | 17 | 6 | 0.0347 |
| ollama:llama3.1:latest | current vs v04 | 72 | 12 | 9 | 0.664 |
| ollama:llama3.1:latest | current vs v05 | 72 | 8 | 16 | 0.152 |
| ollama:qwen3.5:9b | v05 vs v04 | 72 | 22 | 7 | 0.00813 |
| ollama:qwen3.5:9b | current vs v04 | 72 | 25 | 4 | 0.000104 |
| ollama:qwen3.5:9b | current vs v05 | 72 | 11 | 5 | 0.21 |

## Task-level sign test (repeats pooled per task)

Repeats of one task are correlated, so the McNemar p-values above are optimistic. Here each task contributes one paired difference in its number of correct repeats; ties are dropped.

| Model | A vs B | Tasks | A better | B better | Tied | Sign-test p |
|---|---|---|---|---|---|---|
| ollama:llama3.1:latest | v05 vs v04 | 24 | 10 | 3 | 11 | 0.0923 |
| ollama:llama3.1:latest | current vs v04 | 24 | 7 | 5 | 12 | 0.774 |
| ollama:llama3.1:latest | current vs v05 | 24 | 2 | 10 | 12 | 0.0386 |
| ollama:qwen3.5:9b | v05 vs v04 | 24 | 10 | 2 | 12 | 0.0386 |
| ollama:qwen3.5:9b | current vs v04 | 24 | 13 | 3 | 8 | 0.0213 |
| ollama:qwen3.5:9b | current vs v05 | 24 | 8 | 4 | 12 | 0.388 |

## Cost drivers per run

| Model | Surface | Tool calls | Failed calls | LLM calls | Prompt tok | Completion tok |
|---|---|---|---|---|---|---|
| ollama:llama3.1:latest | current | 2.3 | 1.56 | 2.3 | 6,631 | 872 |
| ollama:llama3.1:latest | v04 | 2.4 | 1.22 | 2.3 | 6,811 | 748 |
| ollama:llama3.1:latest | v05 | 2.2 | 1.38 | 2.3 | 6,774 | 847 |
| ollama:qwen3.5:9b | current | 2.1 | 0.61 | 3.1 | 20,560 | 1,719 |
| ollama:qwen3.5:9b | v04 | 1.5 | 0.26 | 2.5 | 16,409 | 1,645 |
| ollama:qwen3.5:9b | v05 | 2.5 | 1.11 | 3.3 | 22,978 | 1,953 |
