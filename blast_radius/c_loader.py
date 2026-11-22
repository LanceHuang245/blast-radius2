"""Runtime MetaPathFinder for modularly compiled mypyc + Cython sqlglot groups.

Each independent compilation group in /app/sqlglot-c/manifest.json only loads its
native .so files when ZERO files belonging to that group were patched by the commit.
Patched files are served by _OverlayFinder (or hot-patched in pure Python), and any
group whose files intersect the patch falls back to pure-Python /app/sqlglot-src.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

from blast_radius import SOURCE_C

MANIFEST_FILE = SOURCE_C / "manifest.json"
_MANIFEST_CACHE: dict | None | object = ...  # sentinel: ... means unloaded

# Tests whose assertions depend on pure-Python sqlglot objects/frames (e.g. Python class
# inheritance on Expression, monkey-patching compiled Dialect/Parser types, Cython stack
# frames/refs, or pure-Python memory layout).
PURE_PYTHON_ONLY_TESTS: frozenset[str] = frozenset({
    "tests/test_generator.py::TestGenerator::test_fallback_function_sql",
    "tests/test_generator.py::TestGenerator::test_fallback_function_var_args_sql",
    "tests/test_expressions.py::TestExpressions::test_custom_expression",
    "tests/test_expressions.py::TestExpressions::test_iter",
    "tests/test_expressions.py::TestExpressions::test_eq",
    "tests/test_expressions.py::TestExpressions::test_expression",
    "tests/test_expressions.py::TestExpressions::test_recursion",
    "tests/dialects/test_dialect.py::TestDialect::test_patch_dialect_parser",
    "tests/dialects/test_dialect.py::TestDialect::test_custom_dialect",
    "tests/test_generator.py::TestGenerator::test_generator_memory",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_skips_unneeded_annotation",
})


def load_c_manifest() -> dict | None:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is ...:
        if MANIFEST_FILE.is_file():
            try:
                _MANIFEST_CACHE = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            except Exception:
                _MANIFEST_CACHE = None
        else:
            _MANIFEST_CACHE = None
    return _MANIFEST_CACHE  # type: ignore[return-value]


def disabled_groups_for(patched_files: dict[str, str | None] | None) -> frozenset[str]:
    """Return C group names that MUST stay pure Python for `patched_files`."""
    manifest = load_c_manifest()
    if manifest is None:
        return frozenset()
    groups = manifest.get("groups", {})
    if not patched_files:
        return frozenset()
    patched = set(patched_files)
    disabled: set[str] = set()
    for gname, info in groups.items():
        if patched & set(info.get("files", ())):
            disabled.add(gname)
    # Package __init__.py edits can re-export symbols consumed by subpackages.
    if "sqlglot/expressions/__init__.py" in patched:
        disabled.update(g for g in ("expr", "parse", "gen") if g in groups)
    if "sqlglot/parsers/__init__.py" in patched and "parse" in groups:
        disabled.add("parse")
    if "sqlglot/generators/__init__.py" in patched and "gen" in groups:
        disabled.add("gen")
    if "sqlglot/optimizer/__init__.py" in patched:
        disabled.update(g for g in groups if g.startswith(("opt_", "cy_")))
    if "sqlglot/executor/__init__.py" in patched:
        disabled.update(g for g in ("exec_table", "cy_python", "cy_context", "cy_env") if g in groups)
    if "expr" in disabled:
        if manifest.get("parse_needs_expr_c") and "parse" in groups:
            disabled.add("parse")
        if manifest.get("gen_needs_expr_c") and "gen" in groups:
            disabled.add("gen")
    return frozenset(disabled)


def phase2_c_bucket(disabled: frozenset[str]) -> str:
    """Group Phase-2 tasks into prewarm buckets for shared COW fork acceleration."""
    if not disabled:
        return "pure_full_c"
    core_exec = {"leaf", "expr", "parse", "gen", "exec_table", "cy_python", "cy_context", "cy_env"}
    if not (disabled & core_exec):
        return "opt_py_c"
    if "parse" in disabled and not (disabled & {"leaf", "expr", "gen", "exec_table", "cy_python", "cy_context", "cy_env"}):
        return "parse_py"
    if "gen" in disabled and not (disabled & {"leaf", "expr", "parse", "exec_table", "cy_python", "cy_context", "cy_env"}):
        return "gen_py"
    return "fallback_c"


class _ModularCFinder(importlib.abc.MetaPathFinder):
    """Serves compiled .so modules for enabled groups ahead of filesystem .py files."""

    def __init__(self, enabled_groups: set[str], manifest: dict):
        self.enabled_groups = frozenset(enabled_groups)
        so_by_mod: dict[str, str] = {}
        groups = manifest.get("groups", {})
        for gname in enabled_groups:
            info = groups.get(gname)
            if not info:
                continue
            so_by_mod.update(info.get("modules", {}))
            so_by_mod.update(info.get("runtime", {}))
        self._so_by_mod = so_by_mod

    def find_spec(self, fullname: str, path=None, target=None):
        so_path = self._so_by_mod.get(fullname)
        if so_path is None:
            return None
        loader = importlib.machinery.ExtensionFileLoader(fullname, so_path)
        return importlib.util.spec_from_file_location(fullname, so_path, loader=loader)


def install_modular_c(
    disabled_groups: set[str] | frozenset[str] = frozenset(),
    after_overlay: bool = True,
) -> frozenset[str]:
    """Install _ModularCFinder for all groups not in `disabled_groups`. Returns enabled group set."""
    # Remove any previously installed _ModularCFinder first.
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _ModularCFinder)]
    manifest = load_c_manifest()
    if manifest is None:
        return frozenset()
    all_groups = set(manifest.get("groups", ()))
    enabled = all_groups - set(disabled_groups)
    if not enabled:
        return frozenset()
    finder = _ModularCFinder(enabled, manifest)
    insert_idx = 0
    if after_overlay:
        for idx, f in enumerate(sys.meta_path):
            if type(f).__name__ == "_OverlayFinder":
                insert_idx = idx + 1
    sys.meta_path.insert(insert_idx, finder)
    return frozenset(enabled)


def compiled_file_set(enabled_groups: frozenset[str] | set[str]) -> set[str]:
    """Return relative sqlglot/*.py file paths currently backed by compiled .so modules."""
    manifest = load_c_manifest()
    if manifest is None or not enabled_groups:
        return set()
    groups = manifest.get("groups", {})
    files: set[str] = set()
    for gname in enabled_groups:
        info = groups.get(gname)
        if info:
            files.update(info.get("files", ()))
    return files
