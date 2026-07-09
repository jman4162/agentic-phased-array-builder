# Trace lab

View APAB agent traces in a browser. One container: Jaeger all-in-one,
which accepts OTLP directly, so no collector is needed.

## Start

```bash
cd lab
docker compose up -d
```

The Jaeger UI is at <http://localhost:16686>; OTLP ingest listens on
4318 (HTTP) and 4317 (gRPC).

## Send a traced run

```bash
pip install "apab[observability,ollama]"

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 APAB_OBSERVABILITY=1 \
    apab run --config apab.yaml
```

Or enable it permanently in `apab.yaml`:

```yaml
observability:
  enabled: true
  otlp_endpoint: http://localhost:4318
```

## Read the trace

In the Jaeger UI, pick the `apab` service and click a trace. Each run
shows one `apab.session` root span with a child `apab.turn` per LLM
turn; under every turn sit the `apab.llm.chat` provider call (token
counts, latency, cost estimate) and one `apab.tool.<name>` span per
tool call (argument hash, status, result summary).

Questions this answers directly:

- Which tool dominates the run's wall time?
- Which tool calls failed, and on which turn?
- How many tokens did each turn consume?
- Did the agent loop repeat the same tool call with the same
  argument hash?

The same spans are written to `<run_dir>/trace.jsonl` in the run
bundle, so runs stay inspectable after the container is gone. See
[docs/observability.md](../docs/observability.md) for the full span
and attribute reference.

The LangGraph pipeline (example 08) and Strands agent (example 07)
also emit spans here: set `OTEL_EXPORTER_OTLP_ENDPOINT` before running
them.

## Stop

```bash
docker compose down
```

## Grafana Tempo instead

If you already run a Grafana stack, point APAB's OTLP exporter at your
[Tempo](https://grafana.com/docs/tempo/latest/) distributor's OTLP
receiver instead — no APAB changes needed beyond the endpoint URL. A
minimal Tempo setup needs three services (OTel Collector, Tempo,
Grafana), which is why this lab defaults to Jaeger.
