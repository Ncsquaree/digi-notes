"""
Proxy loader for `openai`.

This module exists only to allow local development when a stray
`ai/openai.py` would otherwise shadow the installed `openai` package.
On import it attempts to locate a real `openai` package/module on
`sys.path` outside the current project directory and load it. If
found, it replaces the current module in `sys.modules` with the
installed package so the rest of the codebase imports the real
implementation.

If a real `openai` package cannot be located, the module falls back to
raising an informative ImportError so the developer can install the
package into their venv.
"""
import importlib.util
import importlib.machinery
import os
import sys
from types import ModuleType


# If this file is located inside the project's ai/ directory, prefer an
# installed `openai` package on sys.path (site-packages). We search sys.path
# for a candidate `openai` package/module path that is not the current file's
# directory and load it directly to avoid recursive imports.
_this_dir = os.path.dirname(__file__)


def _find_installed_openai():
    candidates = []
    for p in sys.path:
        if not p:
            continue
        try:
            p = os.path.abspath(p)
        except Exception:
            continue
        if p == _this_dir:
            continue
        # look for package directory `openai` or module `openai.py`
        pkg_dir = os.path.join(p, 'openai')
        mod_file = os.path.join(p, 'openai.py')
        if os.path.isdir(pkg_dir):
            candidates.append(pkg_dir)
        elif os.path.isfile(mod_file):
            candidates.append(mod_file)
    return candidates


def _load_module_from_path(path):
    # if path is a package dir, load its __init__.py. If it's a .py, load directly.
    if os.path.isdir(path):
        file_path = os.path.join(path, '__init__.py')
    else:
        file_path = path
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location('openai_installed', file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except Exception:
        return None
    return module


# Attempt to locate and load installed openai package
_candidates = _find_installed_openai()
for _p in _candidates:
    _mod = _load_module_from_path(_p)
    if isinstance(_mod, ModuleType):
        # replace current module in sys.modules so future imports get the real one
        sys.modules['openai'] = _mod
        # also ensure submodule package name works
        if hasattr(_mod, '__path__'):
            sys.modules.setdefault('openai.__init__', _mod)
        break
else:
    # no installed openai found — raise a helpful error when accessed
    raise ImportError("Could not locate an installed 'openai' package. Install 'openai' in your venv (pip install openai) or remove this proxy file.")

symbols the codebase imports during collection: `ChatCompletion` and
an `error` submodule exposing `OpenAIError` and `Timeout` exceptions.

If you have the `openai` package installed in the venv, remove this
shim to use the real implementation.
"""
import sys
import types


class ChatCompletion:
    @staticmethod
    def create(*args, **kwargs):
        raise RuntimeError("openai.ChatCompletion.create not implemented in local shim; install 'openai' package or monkeypatch in tests")


class OpenAIError(Exception):
    pass


class Timeout(Exception):
    pass


# Create a fake submodule `openai.error` so `from openai.error import OpenAIError, Timeout`
# works during import-time. We register it in sys.modules under the proper
# name so normal import machinery resolves it.
_error_mod = types.ModuleType("openai.error")
_error_mod.OpenAIError = OpenAIError
_error_mod.Timeout = Timeout
sys.modules.setdefault("openai.error", _error_mod)

