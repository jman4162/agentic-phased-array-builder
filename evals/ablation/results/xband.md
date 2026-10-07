## Outcomes by model and surface

| Model | Surface | Correct | Confidently wrong | (silently wrong) | Safe failure |
|---|---|---|---|---|---|
| ollama:llama3.1 | current | 1/12 = 8% [1%, 35%] | 7/12 = 58% [32%, 81%] | 0/12 = 0% [0%, 24%] | 4/12 = 33% [14%, 61%] |
| ollama:qwen3.5:9b | current | 6/12 = 50% [25%, 75%] | 6/12 = 50% [25%, 75%] | 5/12 = 42% [19%, 68%] | 0/12 = 0% [0%, 24%] |

## Correct rate by category

| Model | Surface | control | flag |
|---|---|---|---|
| ollama:llama3.1 | current | 0/9 | 1/3 |
| ollama:qwen3.5:9b | current | 6/9 | 0/3 |

## Paired comparisons (same task and repeat)

| Model | A vs B | Pairs | A only correct | B only correct | McNemar p |
|---|---|---|---|---|---|

## Task-level sign test (repeats pooled per task)

Repeats of one task are correlated, so the McNemar p-values above are optimistic. Here each task contributes one paired difference in its number of correct repeats; ties are dropped.

| Model | A vs B | Tasks | A better | B better | Tied | Sign-test p |
|---|---|---|---|---|---|---|

## Cost drivers per run

| Model | Surface | Tool calls | Failed calls | LLM calls | Prompt tok | Completion tok |
|---|---|---|---|---|---|---|
| ollama:llama3.1 | current | 1.8 | 1.08 | 2.3 | 6,633 | 1,037 |
| ollama:qwen3.5:9b | current | 1.5 | 0.08 | 2.5 | 17,675 | 1,779 |
