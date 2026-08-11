# MCP server conventions

The convention for adding an MCP tool surface to a package in this tool family.
It was established here in APAB (18 tools), refined by antenna-cad (7 tools),
and is followed by opensatcom. Follow it exactly when adding `x mcp serve` to a
new package; the point of a convention is that an agent connected to two of
these servers cannot tell which repo it is talking to.

Each item below is written as the rule plus, where it matters, the reason. The
canonical statement of the tool-module rules also lives as a module docstring in
each `tools_*.py`, so the convention travels with the code.

## Decide two things first

- **Is MCP optional?** Default yes: `mcp = ["mcp>=1.26"]` under
  `[project.optional-dependencies]`, a `mcp-test` dependency group folded into
  `dev`, `pytest.importorskip("mcp")` in the tests, and an `ImportError` with an
  install hint. Make `mcp` a core dependency only when the package's whole
  reason to exist is the agent surface (APAB is the one example).
- **Does it need run bundles?** Only if the package *orchestrates* multi-step
  runs. A pure tool surface returns artifact paths and skips provenance
  machinery; the provenance section below applies to orchestrators.

## Module layout

```
src/x/mcp/                  # named mcp/, never agent/ — reserve agent/ for orchestration
  __init__.py               # one-line docstring naming the extra
  server.py                 # _server global, _get_server, get_mcp, run_server
  tools_<domain>.py         # one per bounded domain; a single tools.py is fine under ~8 tools
  resources.py              # optional; x://... URI templates
tests/test_mcp.py           # or test_mcp_server.py + test_mcp_tools_<domain>.py
```

## server.py

- Module docstring states the ordering rule: **the singleton is assigned before
  tool modules are imported**, so their module-level `get_mcp()` re-enters and
  receives the partially built instance instead of recursing.
- `_server: Any = None` at module scope, typed loosely because the class differs
  between MCP SDK generations with the same surface.
- Nested-try SDK import:

  ```python
  try:
      try:
          from mcp.server.fastmcp import FastMCP
      except ImportError:
          from mcp.server import MCPServer as FastMCP
  except ImportError as exc:
      raise ImportError("the MCP server needs the 'mcp' extra: pip install x[mcp]") from exc
  ```

- `_get_server()`: early-return when non-None; assign `_server` **before**
  importing tool modules; wrap each optional tool module in its own
  `try/except ImportError` with a `logger.info` install hint; **import
  `resources.py` here too** if it exists (a resources module that is only
  imported by accident registers by accident).
- `name=` equals the CLI name. `instructions=` is a two-to-four sentence blurb
  listing the tool domains and ending with the payload contract: *"Artifacts
  are returned as file paths, not payloads."*
- `run_server(transport: str = "stdio")`. Add a `create_server(config=...)`
  that stashes config on the singleton only when tools need ambient state;
  otherwise pass paths per call.

## Tool modules

- Header, identical everywhere:

  ```python
  from __future__ import annotations
  import logging
  from typing import Annotated, Any
  from pydantic import Field
  from x.mcp.server import get_mcp

  logger = logging.getLogger(__name__)
  mcp = get_mcp()
  ```

- `@mcp.tool()` (with parens) on `async def`. Tool name is the function name,
  `domain_verb`, domain matching the module suffix: `pattern_compute`,
  `system_evaluate`, `design_drc`, `link_snapshot`.
- **Inputs: flat scalars only.** Every parameter
  `Annotated[T, Field(description="... (units)")]` with units in the
  description; defaults in the signature. Structured inputs degrade to
  `list[dict[str, Any]]` or `list[list[float]]` with the shape spelled out in
  the description, never nested pydantic models. Hoist repeated annotations
  into module-level aliases (`SpecPath`, `OutDir`).
- **Outputs: `-> dict[str, Any]`**, JSON-safe scalars plus artifact paths, a
  `"status"` key on every path. Summarize big arrays (`*_len`, counts) instead
  of returning them.
- **Errors are returned, never raised:**

  ```python
  except Exception as e:
      logger.exception("<tool_name> failed")
      return {"error": str(e), "status": "failed"}
  ```

- Heavy imports inside the function body; `matplotlib.use("Agg")` before
  pyplot. Docstring is one imperative line — it becomes the MCP tool
  description; parameter docs live in `Field(description=...)`, never numpydoc
  sections.
- `logger.info` before the work and after with the headline result.

## Path safety — both checks, shared helpers

- `reject_path_traversal(path) -> Path`: refuse any path containing `..`
  segments. Apply to **every path argument, input and output**, as the first
  statement inside the `try`.
- `validate_path_within(path, root)`: `.resolve()` both and require
  containment. Use whenever the package has a workspace root — traversal
  rejection alone still admits absolute paths outside it.
- Both live in a shared core module (`x/core/workspace.py` here), not as
  private copies in the tools module.
- `Path(out).parent.mkdir(parents=True, exist_ok=True)` before every write.

## CLI wiring

- `x mcp serve`, default transport `stdio`. **Constrain the transport choices**
  (typer enum or argparse `choices=["stdio", "http"]`) and map user-facing
  `http` to the SDK's `streamable-http`; never pass a free-form string through.
- Lazy-import `run_server` inside the command body so `x --help` never imports
  `mcp`.
- README gains the client snippet:

  ```json
  {"mcpServers": {"x": {"command": "x", "args": ["mcp", "serve"]}}}
  ```

## Tests

- `pytest.importorskip("mcp", reason="mcp extra not installed")` at module top
  when MCP is an extra.
- Registration through the **public API** — never the SDK's private tool
  manager:

  ```python
  EXPECTED_TOOLS = {...}  # every tool name, explicitly

  def test_all_tools_registered():
      registered = {t.name for t in asyncio.run(get_mcp().list_tools())}
      assert registered >= EXPECTED_TOOLS
  ```

  Keep the explicit name set so adding a tool forces a test edit.
- Call tool coroutines **directly** for behavior tests (import the function,
  `asyncio.run` or pytest-asyncio) — fast, and the protocol layer is the SDK's
  to test.
- Two contract tests per tool family, always:

  ```python
  def test_errors_returned_not_raised(tmp_path):
      r = asyncio.run(tools.some_tool(str(tmp_path / "missing.yaml")))
      assert r["status"] == "failed" and "error" in r

  def test_path_traversal_rejected():
      r = asyncio.run(tools.some_tool("../../etc/passwd"))
      assert r["status"] == "failed" and "traversal" in r["error"]
  ```

- External binaries get a registered marker and a runtime skip
  (`@pytest.mark.kicad`, `pytest.skip("kicad-cli not installed")`).

## Provenance and observability (orchestrators only)

- Run bundle: `runs/<run_id>/` with `run_id = "%Y%m%dT%H%M%S_" + uuid4().hex[:8]`,
  containing `manifest.json`, `audit.json`, `artifacts/<subdir>/`.
- `manifest.json` minimum keys: `run_id, timestamp, config_hash,
  dependency_versions, artifacts, status`, extended with domain hashes,
  `solver_version`, `provider_name`, `model_name`, `trace_id`, `usage`. Hashes
  are `sha256(json.dumps(d, sort_keys=True, default=str))[:16]`;
  `dependency_versions` walks an explicit package list via
  `importlib.metadata`. Mirror the manifest as a pydantic schema and test that
  a written manifest round-trips through it, so the two cannot drift.
- Manifest writing is wrapped in try/except with `logger.exception` —
  provenance failures never fail the run. Per-tool-call records go to
  `audit.json` (timestamp, tool, trace_id, span_id, arguments,
  result_summary) with a redaction mode, not into the manifest.
- OpenTelemetry is a soft dependency: `_NoopSpan` when disabled, env override
  `X_OBSERVABILITY=1`. Span naming `x.<layer>.<thing>` (`x.pipeline`,
  `x.turn`, `x.tool.<tool_name>`); attributes `x.*`-prefixed **except** LLM
  spans, which use the OpenTelemetry `gen_ai.*` semantic conventions.
- Expose bundles read-only over MCP resources (`x://runs`,
  `x://runs/{run_id}/manifest`, `x://runs/{run_id}/artifacts/{path}`)
  returning JSON strings, with resolve-and-contain checking on the artifact
  route.
