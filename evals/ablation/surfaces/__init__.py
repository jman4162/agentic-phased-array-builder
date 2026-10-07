"""Tool surfaces compared in the ablation.

- ``v04``: the system-tool surface at commit 1eebfcb (options accepted and
  ignored without an error), vendored verbatim in ``tools_system_v04.py``
  with the ``wrappers_pas`` it ran against (``wrappers_pas_v04.py``).
- ``v05``: the surface as released in apab 0.5.0 (commit 5b841f3), vendored
  the same way (``tools_system_v05.py``, ``wrappers_pas_v05.py``), so later
  changes to ``src/`` cannot alter this arm.
- ``v051``: the surface as released in apab 0.5.1 (commit 6484be1), vendored
  the same way. The local sweep's ``current`` arm ran on this surface; 0.5.2
  added ``noise_figure_db`` to ``system_evaluate`` and changed the wrapper's
  speed of light from 3e8 to 299 792 458 m/s (a 0.07% change in spacing in
  wavelengths, about 0.006 dB of array gain).
- ``current``: whatever ``apab.mcp.tools_system`` is installed.

Loading a vendored surface removes the installed system tools from the MCP
singleton and registers the vendored ones under the same names. The only
edit to vendored source is the import path of ``PASSystemEngine``,
redirected to the matching vendored wrapper; the edit is asserted so it
cannot silently fail to apply. All surfaces run on the same installed
phased-array-systems, so the physics is identical and only the tool
interface differs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SURFACES = ("v04", "v05", "v051", "current")
VENDORED = ("v04", "v05", "v051")
SYSTEM_TOOLS = ("system_evaluate", "system_trade_study")

_HERE = Path(__file__).resolve().parent
_WRAPPER_IMPORT = "from apab.system.wrappers_pas import PASSystemEngine"
_loaded: str | None = None


def _load_module(name: str, path: Path, source: str | None = None) -> ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    code = compile(source if source is not None else path.read_text(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def use_surface(name: str) -> None:
    """Make ``name`` the active system-tool surface on the MCP singleton.

    A process holds one surface: switching requires a new process because
    replaced tools are not re-registered. Run each surface in its own process.
    """
    global _loaded
    if name not in SURFACES:
        raise ValueError(f"unknown surface {name!r}; expected one of {SURFACES}")
    if _loaded is not None and _loaded != name:
        raise RuntimeError(f"surface {_loaded!r} already active; use a new process")
    if _loaded == name:
        return

    from apab.mcp import tools_system  # noqa: F401  (registers the installed tools)
    from apab.mcp.server import get_mcp

    if name in VENDORED:
        server = get_mcp()
        for tool in SYSTEM_TOOLS:
            server.remove_tool(tool)

        wrapper_module = f"apab_ablation_wrappers_pas_{name}"
        _load_module(wrapper_module, _HERE / f"wrappers_pas_{name}.py")
        source = (_HERE / f"tools_system_{name}.py").read_text()
        n_imports = source.count(_WRAPPER_IMPORT)
        assert n_imports == 2, f"expected 2 wrapper imports in {name} source, found {n_imports}"
        source = source.replace(_WRAPPER_IMPORT, f"from {wrapper_module} import PASSystemEngine")
        tools_path = _HERE / f"tools_system_{name}.py"
        _load_module(f"apab_ablation_tools_system_{name}", tools_path, source)
    _loaded = name
