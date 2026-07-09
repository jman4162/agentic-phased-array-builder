# Contributing to APAB

## Development Setup

```bash
git clone https://github.com/jman4162/agentic-phased-array-builder.git
cd agentic-phased-array-builder
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ollama]"

# Run tests
pytest tests/ -v

# Linting and type checking
ruff check src/ tests/
mypy src/apab/
```

EdgeFEM requires system-level C++ dependencies:
- **macOS:** `brew install cmake eigen`
- **Ubuntu/Debian:** `sudo apt-get install cmake libeigen3-dev`

## Adding an LLM Provider

APAB discovers providers via the `apab.llm_providers` entry point group. To add a new provider:

1. Create `src/apab/providers/your_provider.py` implementing the `LLMProvider` protocol:

```python
class YourProvider:
    @property
    def name(self) -> str:
        return "your_provider"

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def chat(self, messages, tools=None, **kwargs):
        # Call your API, return {"role": "assistant", "content": ..., "tool_calls": ...}
        ...
```

2. Register it in `pyproject.toml`:

```toml
[project.entry-points."apab.llm_providers"]
your_provider = "apab.providers.your_provider:YourProvider"
```

3. Add the SDK to optional dependencies:

```toml
[project.optional-dependencies]
your_provider = ["your-sdk>=1.0"]
```

See `src/apab/providers/openai.py` or `anthropic.py` for complete examples with tool schema conversion and usage tracking.

## Adding an EM Adapter or Compute Backend

The same entry point pattern applies:

- **EM adapters:** Register under `apab.em_adapters`, implement the `ExternalEMToolAdapter` protocol in `src/apab/emtool/`
- **Compute backends:** Register under `apab.compute_backends`, implement the `ComputeBackend` protocol in `src/apab/compute/`

## Testing Conventions

- Mock external SDKs by patching `sys.modules` (avoids requiring real API keys or installations):

```python
@pytest.fixture()
def _mock_openai():
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        yield mock_openai
```

- Use `@pytest.mark.asyncio` for async MCP tool tests
- Mark slow tests with `@pytest.mark.integration`
- See `tests/test_providers_openai.py` for a complete provider test example

## Code Style

Configured in `pyproject.toml`:
- **Formatter/linter:** ruff (target Python 3.10, 100-char line length)
- **Type checker:** mypy (strict mode)
- **Lint rules:** E, F, W, I, N, UP (pycodestyle, pyflakes, isort, naming, pyupgrade)
