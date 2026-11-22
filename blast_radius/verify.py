"""Dual-phase forked runtime verifier with real-CPU makespan decapitation and 2-core parallelism.

Phase 1 (Clean Fork, before parent imports sqlglot):
    Verifies commits that touch class/module-level code, closures, or test files
    using an in-memory importlib MetaPathFinder + virtual open() overlay.

Phase 2 (Pre-Warmed Fork, after parent imports sqlglot once):
    Verifies commits that only modify existing non-nested function bodies via
    in-place __code__ replacement, and safe literal class-attribute updates.
    Test modules are lazily pre-imported once in the parent and shared via COW.

Soundness invariant: a candidate test is dropped ONLY if it explicitly ran and
passed inside an isolated forked child against the patched source. Any error,
crash, timeout, or unverified test is conservatively retained.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import gc
import importlib
import importlib.abc
import importlib.machinery
import inspect
import io
import marshal
import os
import pickle
import select
import signal
import sys
import textwrap
import time
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from blast_radius import SOURCE, SOURCE_C, log
from blast_radius.c_loader import (
    PURE_PYTHON_ONLY_TESTS,
    disabled_groups_for,
    install_modular_c,
    load_c_manifest,
    phase2_c_bucket,
)
from blast_radius.db_cache import install_db_cache, wrap_duckdb_if_loaded
from blast_radius.impact_map import module_name
from blast_radius.schedule import split_with_cost
from blast_radius.select import Selection
from blast_radius.tpc_verify import (
    TPC_DUCK_BUDGET_S,
    TPC_EST_CPU_MS,
    TPC_REEXEC_BUDGET_S,
    TPC_SUITE_BY_TEST_ID,
    est_tpc_fast_a_ms,
    est_tpc_mode_ms,
    load_tpcds_ckpts_meta,
    select_tpc_mode,
    select_tpcds_ckpt,
    select_tpcds_span,
    tokenizer_edit_position_free,
    tokenizer_edit_touches_max_nodes,
    verify_generator_memory,
    verify_lazy_athena_presto,
    verify_max_nodes,
    verify_opt_tpcds,
    verify_optimized_tpch,
    verify_tpc_test,
)

# Decision thresholds (overridable via env vars for local A/B benchmarking).
MAX_SINGLE_TEST_MS = int(os.environ.get("BR_MAX_SINGLE_MS", "6000"))
MAX_COMMIT_VERIFY_MS = int(os.environ.get("BR_MAX_COMMIT_MS", "6000"))
MIN_GAIN_MS = int(os.environ.get("BR_MIN_GAIN_MS", "300"))
MIN_DECAP_GAIN_MS = int(os.environ.get("BR_MIN_DECAP_GAIN_MS", "550"))
MIN_ROI_RATIO = float(os.environ.get("BR_MIN_ROI_RATIO", "1.20"))
VERIFY_FIXTURE_MAX_MS = int(os.environ.get("BR_FIXTURE_MAX_MS", "0"))
TOTAL_VERIFY_BUDGET_MS = int(os.environ.get("BR_TOTAL_BUDGET_MS", "150000"))
ENABLE_HOTPATCH = os.environ.get("BR_ENABLE_HOTPATCH", "1") != "0"
# Tail verification (Step 2c) limits: how much CPU a child may spend dropping passing tests.
STEP2C_ROUNDS = int(os.environ.get("BR_STEP2C_ROUNDS", "10"))
STEP2C_EST_CAP_MS = int(os.environ.get("BR_STEP2C_EST_MS", "12000"))
STEP2C_CPU_S = float(os.environ.get("BR_STEP2C_CPU_S", "12.0"))
STEP2C_FAIL_ROI = float(os.environ.get("BR_STEP2C_FAIL_ROI", "2.5"))
STEP2C_KEEP_AFTER_FAIL = os.environ.get("BR_STEP2C_KEEP_AFTER_FAIL", "0") != "0"
DECAP_CPU_RATIO = float(os.environ.get("BR_DECAP_CPU_RATIO", "0.85"))
# Weight of a CPU millisecond against a simulated millisecond when choosing what to verify
# (1.0: break-even on the score; below 1: spend CPU to cut simulated time further).
CPU_WEIGHT = float(os.environ.get("BR_CPU_WEIGHT", "1.0"))
PROFILE_TESTS = os.environ.get("BR_PROFILE_TESTS", "0") != "0"
# Pre-import (in the parent) test modules that only ONE task needs: same CPU as importing it in
# that child, and wasted when the child stops early. Modules shared by >= 2 tasks are still warmed.
PREWARM_SINGLE = os.environ.get("BR_PREWARM_SINGLE", "0") != "0"
CHILD_GC_THRESHOLD = int(os.environ.get("BR_CHILD_GC_THRESHOLD", "1000000"))
ENABLE_TPC_FAST = os.environ.get("BR_ENABLE_TPC_FAST", "1") != "0"
ENABLE_TPC_FAST_B = os.environ.get("BR_ENABLE_TPC_FAST_B", "1") != "0"
TPC_PROBE_BUDGET_MS = int(os.environ.get("BR_TPC_PROBE_MS", "300"))
TPC_MIN_GAIN_MS = int(os.environ.get("BR_TPC_MIN_GAIN_MS", "1800"))
MAX_PARALLEL_CHILDREN = int(os.environ.get("BR_PARALLEL", "2"))
SPECIAL_CPU_SCALE = float(os.environ.get("BR_SPECIAL_CPU_SCALE", "1.0"))
FAST_B_MAX_EST_MS = int(os.environ.get("BR_FAST_B_MAX_MS", "6500"))
ENABLE_MYPYC = os.environ.get("BR_MYPYC", "1") != "0" and load_c_manifest() is not None

# Hard wall-clock limits, enforced by the parent (the run itself has 3 minutes, 4.5 in the final).
CHILD_LIMIT_BASE_S = 8.0
CHILD_LIMIT_FACTOR = 3.0
VERIFY_WALL_BUDGET_S = float(os.environ.get("BR_VERIFY_WALL_S", "130"))

OPT_TPCH_TEST_ID = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
OPT_TPCDS_TEST_ID = "tests/test_optimizer.py::TestOptimizer::test_tpcds"
MAX_NODES_TEST_ID = "tests/test_parser.py::TestParser::test_max_nodes"
GEN_MEM_TEST_ID = "tests/test_generator.py::TestGenerator::test_generator_memory"
LAZY_LOAD_TEST_ID = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"

# Tests in TestExecutor that actually use numpy/pandas/TPC fixtures when run via real unittest.
TPC_EXECUTOR_TESTS = frozenset({
    OPT_TPCH_TEST_ID,
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
})

# Tests that inspect fresh import state must never run in Phase 2.
FRESH_IMPORT_TESTS = frozenset({
    LAZY_LOAD_TEST_ID,
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_attribute_access_is_cached",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_import_no_deadlock",
})

# test_lazy_load is verified in <1ms at the start of Phase 1 before any dialects load.
SUBPROCESS_TESTS: frozenset[str] = frozenset()

# Skip redundant pickle round-trips in _LazyFuture when serde/expressions.core are untouched.
_SKIP_POOL_PICKLE = False

LAZY_EXECUTOR_ATTRS = (
    "tpch_conn",
    "tpcds_conn",
    "tpch_tables",
    "tpcds_tables",
    "tpch_sqls",
    "tpcds_sqls",
    "cache",
)


@dataclass
class VerifyTask:
    commit: str
    chosen: set[int]
    selection: Selection
    phase: int  # 1 = clean fork, 2 = pre-warmed fork
    static_cost_ms: int
    est_cpu_ms: int
    expected_gain_ms: int = 0
    tpc_mode: str | None = None
    lazy_pp_nums: set[int] = field(default_factory=set)
    decap_targets: list[int] = field(default_factory=list)
    # Fine-grained case/subTest/block/Validator/TPC filters and their scaled CPU estimates.
    case_filter: dict[int, set[tuple[str, int]]] = field(default_factory=dict)
    subtest_filter: dict[int, tuple[int, int]] = field(default_factory=dict)
    block_filter: dict[int, int] = field(default_factory=dict)
    validator_filter: dict[int, tuple[int, int]] = field(default_factory=dict)
    tpc_query_mask: dict[int, int] = field(default_factory=dict)
    tpc_s2b_query_mask: dict[int, int] = field(default_factory=dict)
    cpu_override: dict[int, int] = field(default_factory=dict)
    disabled_c_groups: frozenset[str] = field(default_factory=frozenset)
    c_bucket: str = "fallback_c"
    c_exec_fast: bool = False
    pure_py_decap: list[int] = field(default_factory=list)
    tok_untouched: bool = False


class _LazyFuture(cf.Future):
    """Evaluates submitted work on .result() so fail-fast subtests abort before running remaining queries."""

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._work = (fn, args, kwargs)
        self._evaluated = False

    def _evaluate(self) -> None:
        if self._evaluated:
            return
        self._evaluated = True
        fn, args, kwargs = self._work
        self._work = None
        try:
            if _SKIP_POOL_PICKLE:
                self.set_result(fn(*args, **kwargs))
            else:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                self.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:
            self.set_exception(error)

    def result(self, timeout=None):
        self._evaluate()
        return super().result(timeout=timeout)


class _InProcessExecutor:
    """Replaces ProcessPoolExecutor in child so workers see in-memory patches with zero spawn cost.

    When serde/expressions.core are patched, arguments and results still round-trip through
    pickle so broken Expression.__reduce__ fails here too. Lazy evaluation on .result() lets
    TestOptimizer.check_file abort on the first failing query instead of running all 500.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*(initargs if _SKIP_POOL_PICKLE else pickle.loads(pickle.dumps(initargs))))
        self._futures: list[_LazyFuture] = []

    def submit(self, fn, /, *args, **kwargs):
        fut = _LazyFuture(fn, args, kwargs)
        self._futures.append(fut)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futs = [self.submit(fn, *args) for args in zip(*iterables)]
        return [f.result() for f in futs]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            for fut in self._futures:
                if not fut._evaluated:
                    fut._evaluate()
        self._futures.clear()
        return False


def _lazy_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
    return set(fs), set()


def _lazy_as_completed(fs, timeout=None):
    return iter(fs)


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, abs_path: str, source_text: str):
        self._abs_path = abs_path
        self._source_text = source_text

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        code = compile(self._source_text, self._abs_path, "exec")
        exec(code, module.__dict__)


class _OverlayFinder(importlib.abc.MetaPathFinder):
    """Serves patched .py modules directly from memory before filesystem finders."""

    def __init__(self, patched_files: dict[str, str | None], root: Path = SOURCE):
        self._modules: dict[str, tuple[str, str | None, bool]] = {}
        for rel_path, text in patched_files.items():
            if rel_path.endswith(".py"):
                mod = module_name(rel_path)
                abs_path = str((root / rel_path).resolve())
                is_pkg = rel_path.endswith("/__init__.py") or rel_path == "__init__.py"
                self._modules[mod] = (abs_path, text, is_pkg)

    def find_spec(self, fullname: str, path=None, target=None):
        entry = self._modules.get(fullname)
        if entry is None:
            return None
        abs_path, text, is_pkg = entry
        if text is None:
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _OverlayLoader(abs_path, text),
            origin=abs_path,
            is_package=is_pkg,
        )
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(abs_path)]
        return spec


def _install_open_overlay(patched_files: dict[str, str | None]) -> None:
    """Redirect builtins.open and io.open reads of patched files to in-memory StringIO/BytesIO."""
    if not patched_files:
        return
    overlay: dict[str, tuple[str, bytes] | None] = {}
    roots = (SOURCE, SOURCE_C) if (SOURCE_C / "sqlglot").is_dir() else (SOURCE,)
    for rel_path, text in patched_files.items():
        val = None if text is None else (text, text.encode("utf-8"))
        for base in roots:
            overlay[os.path.abspath(str(base / rel_path))] = val
            overlay[os.path.realpath(str(base / rel_path))] = val

    orig_open = builtins.open

    def patched_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and "w" not in mode and "a" not in mode and "+" not in mode:
            try:
                resolved = os.path.abspath(os.fsdecode(file))
            except Exception:
                resolved = ""
            if resolved in overlay:
                hit = overlay[resolved]
                if hit is None:
                    raise FileNotFoundError(2, "No such file or directory", resolved)
                text, raw = hit
                if "b" in mode:
                    return io.BytesIO(raw)
                return io.StringIO(text, newline=kwargs.get("newline"))
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = patched_open
    io.open = patched_open


_IMPORT_DIGESTS: dict[str, str] | None = None


class _Deadline(BaseException):
    """Raised by SIGALRM. Not an Exception, so code under test (e.g. a parser loop with
    `except Exception`) cannot swallow it and keep spinning."""


def _raise_timeout(signum, frame):
    raise _Deadline("timed out")


IMPORT_STATE_LIMIT_S = 6.0


def import_state_digests(
    patched_files: dict[str, str],
    names: list[str],
    minimal_mods: list[str] | None = None,
) -> dict[str, str | None]:
    """Import sqlglot modules (minimal closure when safe, else all) with patch in a forked child and digest `names`."""
    gc.freeze()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        payload = marshal.dumps({})
        try:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(30)
            os.chdir(SOURCE)
            sys.path.insert(0, str(SOURCE))
            sys.meta_path.insert(0, _OverlayFinder(patched_files))
            _install_open_overlay(patched_files)
            from blast_radius.import_state import digests_for, import_all_sqlglot, import_minimal_sqlglot

            if minimal_mods is not None:
                import_minimal_sqlglot(minimal_mods)
            else:
                import_all_sqlglot()
            payload = marshal.dumps(digests_for(names))
        except BaseException:
            pass
        finally:
            os.write(w_fd, payload)
            os._exit(0)
    os.close(w_fd)
    chunks = []
    deadline = time.monotonic() + IMPORT_STATE_LIMIT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([r_fd], [], [], remaining)[0]:
            chunks = []  # too slow: whatever the child sends now is not trusted
            break
        chunk = os.read(r_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    _kill_group(pid)
    os.close(r_fd)
    os.waitpid(pid, 0)
    try:
        # A child killed mid-import (timeout, crash, recursion) writes nothing: treat as changed.
        got = marshal.loads(b"".join(chunks))
    except Exception:
        return {}
    return got if isinstance(got, dict) else {}


def check_import_state(patched_files: dict[str, str], owners: list[str]) -> bool:
    """True iff every owner's import-time state digest under the patch equals the pinned one."""
    global _IMPORT_DIGESTS
    if _IMPORT_DIGESTS is None:
        from blast_radius.import_state import STATE_FILE

        _IMPORT_DIGESTS = marshal.loads(STATE_FILE.read_bytes())
    digests = _IMPORT_DIGESTS["digests"]
    min_imports = _IMPORT_DIGESTS.get("min_imports") or {}
    minimal_mods = (
        sorted({m for owner in owners for m in min_imports[owner]})
        if owners and all(owner in min_imports for owner in owners)
        else None
    )
    got = import_state_digests(patched_files, owners, minimal_mods=minimal_mods)
    return all(name in digests and got.get(name) == digests[name] for name in owners)


def _find_unique_code(parent_code: types.CodeType, steps: list[str]) -> types.CodeType | None:
    """Walk nested co_consts to locate the unique compiled CodeType for `qualname`."""
    cur = parent_code
    for depth, name in enumerate(steps):
        matches = [c for c in cur.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        if not matches or (len(matches) > 1 and depth < len(steps) - 1):
            return None
        # Several defs of the function itself are @overload stubs plus the real def, which comes
        # last and is the one bound at runtime (the selector only allows that shape).
        cur = max(matches, key=lambda c: c.co_firstlineno)
    return cur


def _compile_single_func(
    lines: list[str],
    filename: str,
    steps: list[str],
    span: tuple[int, int, bool],
    old_code: types.CodeType,
) -> types.CodeType | None:
    """Compile only the patched function slice (~0.3ms vs ~180ms for full parser.py)."""
    head_lineno, end_lineno, fut_ann = span
    if head_lineno < 1 or end_lineno > len(lines) or head_lineno > end_lineno:
        return None
    try:
        snippet = textwrap.dedent("".join(lines[head_lineno - 1 : end_lineno]))
        hdr: list[str] = []
        indent_prefix = ""
        if fut_ann:
            hdr.append("from __future__ import annotations\n")
        for cls_name in steps[:-1]:
            hdr.append(f"{indent_prefix}class {cls_name}:\n")
            indent_prefix += "    "
        body_src = textwrap.indent(snippet, indent_prefix) if indent_prefix else snippet
        pad_n = max(0, head_lineno - 1 - len(hdr))
        if fut_ann:
            src = hdr[0] + ("\n" * pad_n) + "".join(hdr[1:]) + body_src
        else:
            src = ("\n" * pad_n) + "".join(hdr) + body_src
        mini_code = compile(src, filename, "exec")
        new_code = _find_unique_code(mini_code, steps)
        if new_code is None:
            return None
        if (
            new_code.co_freevars != old_code.co_freevars
            or new_code.co_cellvars != old_code.co_cellvars
        ):
            return None
        return new_code
    except Exception:
        return None


def _apply_code_hotpatch(
    patched_files: dict[str, str],
    targets: list[tuple[str, str]],
    hot_patch_spans: dict[tuple[str, str], tuple[int, int, bool]] | None = None,
    default_hot_targets: list[tuple[str, str, tuple | None, dict | None]] | None = None,
) -> bool:
    """Swap __code__ (and literal defaults) in-place on live sqlglot functions inside the forked Phase-2 child."""
    def_map = {(p, q): (defs, kwdefs) for p, q, defs, kwdefs in (default_hot_targets or ())}
    by_path: dict[str, list[str]] = {}
    for rel_path, qualname in targets:
        by_path.setdefault(rel_path, []).append(qualname)

    for rel_path, qualnames in by_path.items():
        text = patched_files.get(rel_path)
        if text is None:
            return False
        mod_name = module_name(rel_path)
        mod = importlib.import_module(mod_name)
        filename = str((Path(os.getcwd()) / rel_path).resolve())
        lines: list[str] | None = None
        mod_code: types.CodeType | None = None
        for qualname in qualnames:
            steps = qualname.split(".")
            container: object = mod
            for part in steps[:-1]:
                container = container.__dict__.get(part) if hasattr(container, "__dict__") else None
                if container is None:
                    return False
            attr = steps[-1]
            raw = container.__dict__.get(attr) if hasattr(container, "__dict__") else None
            if isinstance(raw, (staticmethod, classmethod)):
                fn = raw.__func__
            elif isinstance(raw, property):
                fn = raw.fget
            elif inspect.isfunction(raw):
                fn = raw
            else:
                return False
            if not inspect.isfunction(fn):
                return False
            old_code = fn.__code__
            new_code: types.CodeType | None = None
            span = hot_patch_spans.get((rel_path, qualname)) if hot_patch_spans else None
            if span is not None:
                if lines is None:
                    lines = text.splitlines(keepends=True)
                new_code = _compile_single_func(lines, filename, steps, span, old_code)
            if new_code is None:
                if mod_code is None:
                    mod_code = compile(text, filename, "exec")
                new_code = _find_unique_code(mod_code, steps)
            if new_code is None:
                return False
            if (
                old_code.co_freevars != new_code.co_freevars
                or old_code.co_cellvars != new_code.co_cellvars
                or len(fn.__closure__ or ()) != len(new_code.co_freevars)
            ):
                return False
            fn.__code__ = new_code
            if (rel_path, qualname) in def_map:
                new_defs, new_kwdefs = def_map[(rel_path, qualname)]
                fn.__defaults__ = new_defs
                fn.__kwdefaults__ = new_kwdefs
    return True


def _apply_attr_hotpatch(targets: list[tuple[str, str, str, object]]) -> bool:
    """Update non-import-time literal class attributes in-place inside the forked Phase-2 child."""
    for rel_path, cls_qual, attr_name, lit_val in targets:
        mod = importlib.import_module(module_name(rel_path))
        container: object = mod
        for part in cls_qual.split("."):
            container = container.__dict__.get(part) if hasattr(container, "__dict__") else None
            if container is None:
                return False
        if not isinstance(container, type) or attr_name not in container.__dict__:
            return False
        setattr(container, attr_name, lit_val)
    return True


def _closure_refs_target(fn, old_val) -> bool:
    if not callable(fn):
        return False
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            cval = cell.cell_contents
            if cval is old_val or cval == old_val:
                return True
        except Exception:
            pass
    return False


def _apply_dict_hotpatch(
    targets: list[tuple[str, str, str, list[str], list[tuple[str, str]]]],
) -> bool:
    """Update non-import-time tracked dict entries (and safely copied subclass dicts) in Phase-2 child."""
    updates: list[tuple[type, str, dict]] = []
    cache_clears: list[type] = []
    for rel_path, cls_qual, attr_name, removed_keys, set_items in targets:
        mod = importlib.import_module(module_name(rel_path))
        container: object = mod
        for part in cls_qual.split("."):
            container = container.__dict__.get(part) if hasattr(container, "__dict__") else None
            if container is None:
                return False
        if not isinstance(container, type):
            return False
        cur = container.__dict__.get(attr_name)
        if not isinstance(cur, dict):
            return False
        new_d = dict(cur)
        filename = str((Path(os.getcwd()) / rel_path).resolve())
        eval_removed = []
        eval_set = []
        try:
            for k_src in removed_keys:
                k = eval(compile(k_src, filename, "eval"), mod.__dict__)
                if k not in new_d:
                    return False
                eval_removed.append(k)
                del new_d[k]
            for k_src, v_src in set_items:
                k = eval(compile(k_src, filename, "eval"), mod.__dict__)
                v = eval(compile(v_src, filename, "eval"), mod.__dict__)
                eval_set.append((k, v))
                new_d[k] = v
        except Exception:
            return False

        updates.append((container, attr_name, new_d))
        if attr_name == "TRANSFORMS":
            cache_clears.append(container)

        # Propagate changed/added/removed entries to sqlglot subclasses that copied parent's dict.
        subs: list[type] = []
        seen_subs: set[type] = {container}
        queue = list(container.__subclasses__())
        while queue:
            sub = queue.pop(0)
            if sub in seen_subs or not sub.__module__.startswith("sqlglot"):
                continue
            seen_subs.add(sub)
            subs.append(sub)
            queue.extend(sub.__subclasses__())

        for sub in subs:
            if attr_name == "TRANSFORMS":
                cache_clears.append(sub)
            if attr_name not in sub.__dict__:
                continue
            sub_d = sub.__dict__[attr_name]
            if sub_d is cur:
                updates.append((sub, attr_name, new_d))
                continue
            if not isinstance(sub_d, dict):
                return False
            new_sub = dict(sub_d)
            changed = False
            try:
                for k in eval_removed:
                    if k in new_sub:
                        if new_sub[k] == cur[k]:
                            del new_sub[k]
                            changed = True
                        elif _closure_refs_target(new_sub[k], cur[k]):
                            return False
                for k, v in eval_set:
                    if k in cur:
                        if k in new_sub:
                            if new_sub[k] == cur[k]:
                                new_sub[k] = v
                                changed = True
                            elif _closure_refs_target(new_sub[k], cur[k]):
                                return False
                    elif k not in new_sub:
                        new_sub[k] = v
                        changed = True
            except Exception:
                return False
            if changed:
                updates.append((sub, attr_name, new_sub))

    try:
        for cls_obj, attr_name, dict_val in updates:
            setattr(cls_obj, attr_name, dict_val)
    except TypeError:
        return False
    gen_mod = sys.modules.get("sqlglot.generator")
    if gen_mod is not None and hasattr(gen_mod, "_DISPATCH_CACHE"):
        dcache = gen_mod._DISPATCH_CACHE
        for cls_obj in cache_clears:
            dcache.pop(cls_obj, None)
    return True


def _install_lazy_executor_setup(cls: type) -> None:
    """Defer TestExecutor's heavy TPC-H/TPC-DS CSV + DuckDB setup until actually accessed."""
    orig_cm = cls.__dict__.get("setUpClass")
    if not isinstance(orig_cm, classmethod):
        return
    real_setup = orig_cm.__func__
    initialized = [False]

    def ensure_initialized() -> None:
        if initialized[0]:
            return
        initialized[0] = True
        for attr_name in LAZY_EXECUTOR_ATTRS:
            if attr_name in cls.__dict__:
                delattr(cls, attr_name)
        real_setup(cls)

    class _LazyAttr:
        __slots__ = ("_name",)

        def __init__(self, name: str):
            self._name = name

        def __get__(self, obj, objtype=None):
            ensure_initialized()
            return getattr(cls, self._name)

    cls.setUpClass = classmethod(lambda c: None)
    for attr_name in LAZY_EXECUTOR_ATTRS:
        setattr(cls, attr_name, _LazyAttr(attr_name))


def _install_lazy_optimizer_setup(cls: type) -> None:
    """Defer TestOptimizer's DuckDB table creation until a test actually accesses self.conn."""
    if getattr(cls, "_br_lazy_opt_setup", False):
        return
    orig_cm = cls.__dict__.get("setUpClass")
    if not isinstance(orig_cm, classmethod):
        return
    cls._br_lazy_opt_setup = True
    real_setup = orig_cm.__func__
    conn_initialized = [False]

    def ensure_conn() -> None:
        if conn_initialized[0]:
            return
        conn_initialized[0] = True
        if "conn" in cls.__dict__:
            delattr(cls, "conn")
        real_setup(cls)

    class _LazyConn:
        def __get__(self, obj, objtype=None):
            ensure_conn()
            return getattr(cls, "conn")

    def light_setup(c) -> None:
        import sqlglot
        from sqlglot.schema import MappingSchema

        c.pool = _InProcessExecutor()
        sqlglot.schema = MappingSchema()

    cls.setUpClass = classmethod(light_setup)
    setattr(cls, "conn", _LazyConn())


def _install_lazy_data_libs() -> None:
    """Defer importing numpy and pandas until actually accessed by a test.

    Both tests/test_executor.py and tests/test_optimizer.py import pandas.testing
    at module top level, even though most tests never call assert_frame_equal.
    """
    if "numpy" in sys.modules or "pandas" in sys.modules:
        return

    stub_names = ("numpy", "pandas", "pandas.testing")

    def _load_real(name: str) -> types.ModuleType:
        for s in stub_names:
            mod = sys.modules.get(s)
            if getattr(mod, "_br_lazy_stub", False):
                sys.modules.pop(s, None)
        real = importlib.import_module(name)
        for mod_key in ("tests.test_executor", "tests.test_optimizer"):
            tmod = sys.modules.get(mod_key)
            if tmod is not None:
                if name == "numpy" and getattr(getattr(tmod, "np", None), "_br_lazy_stub", False):
                    tmod.np = real
                elif name == "pandas" and getattr(getattr(tmod, "pd", None), "_br_lazy_stub", False):
                    tmod.pd = real
                elif name == "pandas.testing":
                    tmod.assert_frame_equal = real.assert_frame_equal
        return real

    class _LazyModule(types.ModuleType):
        _br_lazy_stub = True

        def __getattr__(self, attr: str):
            return getattr(_load_real(self.__name__), attr)

    def _lazy_assert_frame_equal(*args, **kwargs):
        real_testing = _load_real("pandas.testing")
        return real_testing.assert_frame_equal(*args, **kwargs)

    np_stub = _LazyModule("numpy")
    pd_stub = _LazyModule("pandas")
    pd_testing_stub = _LazyModule("pandas.testing")
    pd_testing_stub.assert_frame_equal = _lazy_assert_frame_equal
    pd_stub.testing = pd_testing_stub
    sys.modules["numpy"] = np_stub
    sys.modules["pandas"] = pd_stub
    sys.modules["pandas.testing"] = pd_testing_stub


def _patch_pool_references() -> None:
    """Ensure concurrent.futures and imported test modules use lazy fail-fast _InProcessExecutor."""
    cf.ProcessPoolExecutor = _InProcessExecutor
    for mod_name in ("tests.test_optimizer", "tests.test_executor"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "ProcessPoolExecutor"):
                setattr(mod, "ProcessPoolExecutor", _InProcessExecutor)
            if mod_name == "tests.test_optimizer":
                setattr(mod, "wait", _lazy_wait)
                setattr(mod, "as_completed", _lazy_as_completed)


def _install_fail_fast_subtest() -> None:
    """Abort a test immediately on the first failing subtest (while honoring SkipTest)."""

    @contextmanager
    def _ff_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        try:
            yield
        except unittest.SkipTest:
            pass

    unittest.TestCase.subTest = _ff_subtest


_CASE_STATE: dict = {
    "allowed": None,
    "in_check": False,
    "subtest_mask": None,
    "subtest_stride": 1,
    "subtest_idx": 0,
    "block_mask": None,
    "val_mask": None,
    "val_stride": 1,
    "val_idx": 0,
    "val_depth": 0,
}


def _br_subtest_ok() -> bool:
    mask = _CASE_STATE["subtest_mask"]
    if mask is None:
        return True
    idx = _CASE_STATE["subtest_idx"]
    _CASE_STATE["subtest_idx"] = idx + 1
    return bool(mask >> (idx // _CASE_STATE["subtest_stride"]) & 1)


def _br_block_ok(idx: int) -> bool:
    mask = _CASE_STATE["block_mask"]
    return mask is None or bool(mask >> idx & 1)


def _install_case_filter(mod) -> None:
    """While a fixture-mapped test runs, let TestOptimizer.check_file see only the allowed
    (fixture file, index) cases (see blast_radius.precompile_cases)."""
    if getattr(mod, "_br_case_filter", False):
        return
    mod._br_case_filter = True
    orig_loader = mod.load_sql_fixture_pairs

    def loader(filename):
        allowed = _CASE_STATE["allowed"] if _CASE_STATE["in_check"] else None
        for index, item in enumerate(orig_loader(filename)):
            if allowed is None or (filename, index) in allowed:
                yield item

    mod.load_sql_fixture_pairs = loader
    orig_check_file = mod.TestOptimizer.check_file

    def check_file(self, *args, **kwargs):
        _CASE_STATE["in_check"] = True
        try:
            return orig_check_file(self, *args, **kwargs)
        finally:
            _CASE_STATE["in_check"] = False

    mod.TestOptimizer.check_file = check_file


def _install_subtest_filter(mod, cls: type, meth_name: str) -> None:
    """Lazily wrap `cls.<meth_name>` so its top-level `with self.subTest(...):` blocks can filter by bitmask."""
    fn = cls.__dict__.get(meth_name)
    if fn is not None and not getattr(fn, "_br_subtest_wrapped", False):
        from blast_radius.precompile_cases import transform_subtest_method

        transform_subtest_method(mod, cls, meth_name, _br_subtest_ok)


def _install_block_filter(mod, cls: type, meth_name: str) -> None:
    """Lazily wrap `cls.<meth_name>` so independent top-level statement blocks can filter by bitmask."""
    fn = cls.__dict__.get(meth_name)
    if fn is not None and not getattr(fn, "_br_block_count", 0):
        from blast_radius.precompile_cases import transform_block_method

        transform_block_method(mod, cls, meth_name, _br_block_ok)


def _install_validator_filter() -> None:
    """Wrap Validator.validate_identity/validate_all/validate_transpile once to skip unreached calls."""
    from tests.dialects.test_dialect import Validator

    if getattr(Validator, "_br_val_filter", False):
        return
    Validator._br_val_filter = True
    for attr in ("validate_identity", "validate_all", "validate_transpile"):
        orig = getattr(Validator, attr)

        def make_wrapped(fn):
            def wrapped(self, *args, **kwargs):
                mask = _CASE_STATE["val_mask"]
                if mask is None or _CASE_STATE["val_depth"] > 0:
                    return fn(self, *args, **kwargs)
                idx = _CASE_STATE["val_idx"]
                _CASE_STATE["val_idx"] = idx + 1
                if not (mask >> (idx // _CASE_STATE["val_stride"]) & 1):
                    return None
                _CASE_STATE["val_depth"] += 1
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    _CASE_STATE["val_depth"] -= 1

            return wrapped

        setattr(Validator, attr, make_wrapped(orig))


def _can_fast_opt_tpch(sel: Selection) -> bool:
    return not any(
        p in ("tests/test_executor.py", "tests/helpers.py") or p.startswith("tests/fixtures/optimizer/tpc-h/")
        for p in sel.patched_files
    )


def _can_fast_max_nodes(sel: Selection) -> bool:
    return "tests/test_parser.py" not in sel.patched_files


def _can_fast_gen_mem(sel: Selection) -> tuple[bool, bool]:
    """Return (can_fast, is_scoped_only)."""
    if any(
        p in ("tests/test_generator.py", "tests/helpers.py", "tests/__init__.py", "tests/conftest.py")
        for p in sel.patched_files
    ):
        return False, False
    core_touched = any(
        p.startswith("sqlglot/")
        and (
            not p.startswith(("sqlglot/dialects/", "sqlglot/generators/", "sqlglot/parsers/", "sqlglot/typing/"))
            or p
            in (
                "sqlglot/dialects/dialect.py",
                "sqlglot/dialects/__init__.py",
                "sqlglot/generators/__init__.py",
                "sqlglot/parsers/base.py",
                "sqlglot/typing/__init__.py",
            )
        )
        for p in sel.patched_files
    )
    return True, not core_touched


def _est_test_cpu_ms(
    num: int,
    tid: str,
    sel: Selection,
    tpc_mode: str | None,
    cpu_by_number: dict[int, int],
    tpc_query_mask: dict[int, int] | None = None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
    c_exec_fast: bool = False,
    tok_untouched: bool | None = None,
    leaf_compiled: bool = False,
    tpc_c_ms: dict[str, int] | None = None,
) -> int:
    """Return the CPU estimate (ms) for verifying test `num` on `sel`, safety factor included."""
    special = _special_cpu_ms(
        num,
        tid,
        sel,
        tpc_mode,
        tpc_query_mask,
        tpc_s2b_query_mask,
        c_exec_fast=c_exec_fast,
        tok_untouched=tok_untouched,
        leaf_compiled=leaf_compiled,
        tpc_c_ms=tpc_c_ms,
    )
    # cpu_by_number already carries BR_CPU_SCALE (applied in __main__); dedicated verifiers'
    # estimates are measured end to end and use their own factor.
    return cpu_by_number[num] if special is None else round(special * SPECIAL_CPU_SCALE)


def _special_cpu_ms(
    num: int,
    tid: str,
    sel: Selection,
    tpc_mode: str | None,
    tpc_query_mask: dict[int, int] | None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
    c_exec_fast: bool = False,
    tok_untouched: bool | None = None,
    leaf_compiled: bool = False,
    tpc_c_ms: dict[str, int] | None = None,
) -> int | None:
    """Raw CPU estimate for tests with a dedicated fast verifier, else None."""
    if tid == LAZY_LOAD_TEST_ID and "tests/dialects/test_dialect.py" not in sel.patched_files:
        return 5
    if tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
        suite = TPC_SUITE_BY_TEST_ID[tid]
        q_mask = tpc_query_mask.get(num) if tpc_query_mask is not None else None
        s2b_mask = tpc_s2b_query_mask.get(num) if tpc_s2b_query_mask is not None else None
        total_q = 99 if suite == "tpcds" else 22
        est = est_tpc_mode_ms(
            suite,
            tpc_mode,
            sel.patched_files,
            q_mask.bit_count() if q_mask is not None else None,
            s2b_mask.bit_count() if s2b_mask is not None else None,
            total_q,
            edited_spans=sel.edited_spans,
            s2b_mask=s2b_mask,
            s1_mask=q_mask,
            c_exec_fast=c_exec_fast,
        )
        if suite == "tpcds" and tpc_mode == "fast_a" and c_exec_fast and tpc_c_ms and "exec_fast_a" in tpc_c_ms:
            if q_mask is None:
                est = min(est, tpc_c_ms["exec_fast_a"])
        return est
    if tid == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(sel):
        return 75 if c_exec_fast else 100
    if tid == MAX_NODES_TEST_ID and _can_fast_max_nodes(sel):
        is_untouched = (not tokenizer_edit_touches_max_nodes(sel)) if tok_untouched is None else tok_untouched
        if is_untouched or tokenizer_edit_position_free(sel):
            return 45
        if leaf_compiled:
            return 1500
        return 16500
    if tid == GEN_MEM_TEST_ID:
        can_fast, is_scoped = _can_fast_gen_mem(sel)
        if can_fast:
            return 450 if is_scoped else 550
    return None


def _lb_cost(total: int, biggest: int, count: int) -> int:
    """Lower bound of a commit's cost for tests summing `total` ms (largest `biggest`)."""
    if count <= 0 or total <= 0:
        return 0
    return min(max(biggest, -(-total // k)) + 2000 * k for k in range(1, min(8, count) + 1))


def _best_drop_prefix(
    pool: set[int],
    candidates,
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    min_gain: int,
    limit: int = 0,
    id_by_number: dict[int, str] | None = None,
    per_mod_ms: int = 0,
    min_roi: float = MIN_ROI_RATIO,
) -> list[int]:
    """Candidates worth verifying to drop from `pool`: scans ALL candidates across 3 orderings
    (ratio, module-grouped, descending sim) returning the prefix with best net gain."""
    cand_set = set(candidates)
    if not cand_set:
        return []
    # When pool cannot be emptied to 0 workers, 0ms tests can never reduce simulated cost.
    can_empty = pool <= cand_set
    if not can_empty:
        cand_list = [n for n in cand_set if ms_by_number[n] > 0]
    else:
        cand_list = list(cand_set)
    if not cand_list:
        return []

    def _ratio_key(n: int):
        return (
            1 if ms_by_number[n] == 0 else 0,
            cpu_by_number[n] / max(ms_by_number[n], 1),
            -ms_by_number[n],
            n,
        )

    order_ratio = sorted(cand_list, key=_ratio_key)
    orderings: list[list[int]] = [order_ratio]

    if per_mod_ms and id_by_number is not None:
        by_mod: dict[str, list[int]] = {}
        for n in order_ratio:
            by_mod.setdefault(id_by_number[n].split("::", 1)[0], []).append(n)
        if len(by_mod) > 1:
            mod_keys: list[tuple] = []
            for mod, nums in by_mod.items():
                s_ms = sum(ms_by_number[n] for n in nums)
                s_cpu = sum(cpu_by_number[n] for n in nums) + per_mod_ms
                mod_keys.append((1 if s_ms == 0 else 0, s_cpu / max(s_ms, 1), -s_ms, mod))
            mod_keys.sort()
            order_mod = [n for _, _, _, mod in mod_keys for n in by_mod[mod]]
            if order_mod != order_ratio:
                orderings.append(order_mod)

    order_desc = sorted(cand_list, key=lambda n: (-ms_by_number[n], cpu_by_number[n], n))
    if order_desc != order_ratio:
        orderings.append(order_desc)

    desc = sorted(pool, key=lambda n: (-ms_by_number[n], n))
    pool_total = sum(ms_by_number[n] for n in pool)
    pool_count = len(pool)
    current = _lb_cost(pool_total, ms_by_number[desc[0]] if desc else 0, pool_count)

    best_net = min_gain - 1
    best: list[int] = []
    for order in orderings:
        scan = order if limit <= 0 else order[: max(limit, len(order))]
        removed: set[int] = set()
        seen_mods: set[str] = set()
        prefix: list[int] = []
        total, count, spent, top = pool_total, pool_count, 0, 0
        for num in scan:
            removed.add(num)
            prefix.append(num)
            total -= ms_by_number[num]
            count -= 1
            spent += cpu_by_number[num]
            if per_mod_ms and id_by_number is not None:
                mod_file = id_by_number[num].split("::", 1)[0]
                if mod_file not in seen_mods:
                    seen_mods.add(mod_file)
                    spent += per_mod_ms
            while top < len(desc) and desc[top] in removed:
                top += 1
            saved = current - _lb_cost(total, ms_by_number[desc[top]] if top < len(desc) else 0, count)
            net = saved - spent * CPU_WEIGHT
            if net > best_net and saved >= spent * min_roi * CPU_WEIGHT:
                best_net, best = net, list(prefix)
    return best


def _has_decap_prefix_roi(
    cur_pool: set[int],
    elig_rem: list[int],
    failed: set[int],
    sel: Selection,
    tpc_mode: str | None,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    tpc_query_mask: dict[int, int] | None = None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
    c_exec_fast: bool = False,
    tok_untouched: bool | None = None,
    leaf_compiled: bool = False,
    tpc_c_ms: dict[str, int] | None = None,
) -> bool:
    """Check if the tallest remaining decap target `elig_rem[0]` belongs to a net-profitable prefix."""
    if not elig_rem:
        return False
    num = elig_rem[0]
    t_cpu = _est_test_cpu_ms(
        num,
        id_by_number[num],
        sel,
        tpc_mode,
        cpu_by_number,
        tpc_query_mask,
        tpc_s2b_query_mask,
        c_exec_fast=c_exec_fast,
        tok_untouched=tok_untouched,
        leaf_compiled=leaf_compiled,
        tpc_c_ms=tpc_c_ms,
    )
    cur_c, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
    min_req = MIN_DECAP_GAIN_MS if failed else MIN_GAIN_MS
    trial = set(cur_pool)
    acc_cpu = 0
    for cand in elig_rem:
        trial.discard(cand)
        acc_cpu += _est_test_cpu_ms(
            cand,
            id_by_number[cand],
            sel,
            tpc_mode,
            cpu_by_number,
            tpc_query_mask,
            tpc_s2b_query_mask,
            c_exec_fast=c_exec_fast,
            tok_untouched=tok_untouched,
            leaf_compiled=leaf_compiled,
            tpc_c_ms=tpc_c_ms,
        )
        after_c, _ = split_with_cost([(ms_by_number[n], n) for n in trial])
        if (cur_c - after_c) - acc_cpu >= min_req:
            with_num_c, _ = split_with_cost([(ms_by_number[n], n) for n in (trial | {num})])
            if with_num_c - after_c > t_cpu:
                return True
    return False


def _start_child_verification(
    task: VerifyTask,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    out_fd: int | None,
) -> tuple[int, int]:
    """Fork an isolated child to verify candidate tests; return (pid, read_pipe_fd)."""
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid != 0:
        os.close(w_fd)
        return pid, r_fd

    os.close(r_fd)
    # Own process group: a timeout kill must also reach subprocesses the tests spawned, or they
    # keep burning CPU and hold the result pipe open (the parent's drain would block on them).
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    if out_fd is not None:
        try:
            os.close(out_fd)
        except OSError:
            pass
    try:
        global _SKIP_POOL_PICKLE
        gc.enable()
        if CHILD_GC_THRESHOLD > 0:
            # Short-lived child: AST parent links make every parse/optimize churn reference cycles,
            # and generation-0 collections were a sizable share of verification CPU. Explicit
            # gc.collect() calls in tests still run.
            gc.set_threshold(CHILD_GC_THRESHOLD, 50, 100)
        cpu_by_number = {**cpu_by_number, **task.cpu_override}
        _SKIP_POOL_PICKLE = not any(
            p in ("sqlglot/serde.py", "sqlglot/expressions/core.py") for p in task.selection.patched_files
        )
        root = SOURCE
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _patch_pool_references()
        _install_fail_fast_subtest()

        timeouts = [0]
        import_hung = [False]

        def _alarm_handler(_signum, _frame):
            timeouts[0] += 1
            raise _Deadline("test timed out")

        # Installed before any patched import: a mutant that loops at import time must raise,
        # not hang the child until the parent's much longer backstop kill.
        signal.signal(signal.SIGALRM, _alarm_handler)

        manifest = load_c_manifest() if ENABLE_MYPYC else None
        all_c_groups = frozenset(manifest.get("groups", ())) if manifest else frozenset()
        c_enabled = bool(all_c_groups - task.disabled_c_groups)
        leaf_compiled = manifest is not None and ("leaf" not in task.disabled_c_groups)

        lazy_load_ok: bool | None = None
        if task.phase == 1:
            if "sqlglot" in sys.modules:
                for k in [m for m in sys.modules if m == "sqlglot" or m.startswith(("sqlglot.", "tests."))]:
                    sys.modules.pop(k, None)
            sys.meta_path.insert(0, _OverlayFinder(task.selection.patched_files, root=root))
            if c_enabled:
                install_modular_c(task.disabled_c_groups, after_overlay=True)
            install_db_cache()
            _install_open_overlay(task.selection.patched_files)
            if (
                "tests/dialects/test_dialect.py" not in task.selection.patched_files
                and any(id_by_number[n] == LAZY_LOAD_TEST_ID for n in task.chosen)
            ):
                signal.alarm(6)
                try:
                    import sqlglot as _sg

                    # Check both lazy dialect loading AND that test_dialect.py imports cleanly
                    # (collection fails in pytest if a dialect imported by test_dialect crashes).
                    lazy_ok = len(_sg.Dialect._classes) == 1
                    if lazy_ok:
                        importlib.import_module("tests.dialects.test_dialect")
                    lazy_load_ok = lazy_ok
                except BaseException:
                    lazy_load_ok = False
                    # Importing the patched dialects hangs: every test would hang the same way.
                    import_hung[0] = timeouts[0] > 0
                finally:
                    signal.alarm(0)
        else:
            install_db_cache()
            wrap_duckdb_if_loaded()
            _install_open_overlay(task.selection.patched_files)
            if task.selection.hot_patch_targets:
                if not _apply_code_hotpatch(
                    task.selection.patched_files,
                    task.selection.hot_patch_targets,
                    task.selection.hot_patch_spans,
                    task.selection.default_hot_targets,
                ):
                    os.write(w_fd, b"FAIL_PATCH")
                    os.close(w_fd)
                    os._exit(0)
            if task.selection.attr_hot_targets:
                if not _apply_attr_hotpatch(task.selection.attr_hot_targets):
                    os.write(w_fd, b"FAIL_PATCH")
                    os.close(w_fd)
                    os._exit(0)
            if task.selection.dict_hot_targets:
                if not _apply_dict_hotpatch(task.selection.dict_hot_targets):
                    os.write(w_fd, b"FAIL_PATCH")
                    os.close(w_fd)
                    os._exit(0)

        failed: set[int] = set()
        passed: set[int] = set()
        loaded_classes: dict[tuple[str, str], type] = {}
        loaded_mods: dict[str, types.ModuleType] = {}
        setup_done: set[type] = set()
        sel = task.selection
        executor_patched = "tests/test_executor.py" in sel.patched_files
        needs_real_tpc = any(
            _real_tpc_possible(task.tpc_mode, sel, n, id_by_number, ms_by_number, cpu_by_number)
            for n in task.chosen
        )
        if not executor_patched and not needs_real_tpc:
            _install_lazy_data_libs()

        def _run_unittest(num: int) -> bool:
            tid = id_by_number[num]
            parts = tid.split("::")
            if len(parts) != 3:
                return False
            file_path, cls_name, meth_name = parts
            mod_name = file_path[:-3].replace("/", ".")
            key = (mod_name, cls_name)

            ok = False
            est_s = max(ms_by_number[num], cpu_by_number[num]) / 1000
            timeout_s = max(5, int(est_s * 5) + 4)
            signal.alarm(timeout_s)
            try:
                if key not in loaded_classes:
                    mod = importlib.import_module(mod_name)
                    wrap_duckdb_if_loaded()
                    loaded_mods[mod_name] = mod
                    if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                        _patch_pool_references()
                    if mod_name == "tests.test_optimizer":
                        _install_case_filter(mod)
                    cls = getattr(mod, cls_name)
                    if cls_name == "TestExecutor" and not executor_patched:
                        _install_lazy_executor_setup(cls)
                    elif cls_name == "TestOptimizer" and "tests/test_optimizer.py" not in sel.patched_files:
                        _install_lazy_optimizer_setup(cls)
                    loaded_classes[key] = cls
                    if cls not in setup_done:
                        cls.setUpClass()
                        wrap_duckdb_if_loaded()
                        setup_done.add(cls)
                cls = loaded_classes[key]
                sub_spec = task.subtest_filter.get(num)
                blk_mask = task.block_filter.get(num)
                val_spec = task.validator_filter.get(num)
                if sub_spec is not None:
                    _install_subtest_filter(loaded_mods[mod_name], cls, meth_name)
                if blk_mask is not None:
                    _install_block_filter(loaded_mods[mod_name], cls, meth_name)
                if val_spec is not None:
                    _install_validator_filter()
                res = unittest.TestResult()
                _CASE_STATE["allowed"] = task.case_filter.get(num)
                if sub_spec is not None:
                    _CASE_STATE["subtest_mask"], _CASE_STATE["subtest_stride"] = sub_spec
                else:
                    _CASE_STATE["subtest_mask"], _CASE_STATE["subtest_stride"] = None, 1
                _CASE_STATE["subtest_idx"] = 0
                _CASE_STATE["block_mask"] = blk_mask
                if val_spec is not None:
                    _CASE_STATE["val_mask"], _CASE_STATE["val_stride"] = val_spec
                else:
                    _CASE_STATE["val_mask"], _CASE_STATE["val_stride"] = None, 1
                _CASE_STATE["val_idx"] = 0
                _CASE_STATE["val_depth"] = 0
                cls(meth_name).run(res)
                ok = res.wasSuccessful()
            except BaseException:
                ok = False
            finally:
                _CASE_STATE["allowed"] = None
                _CASE_STATE["subtest_mask"] = None
                _CASE_STATE["block_mask"] = None
                _CASE_STATE["val_mask"] = None
                _CASE_STATE["val_depth"] = 0
                signal.alarm(0)
            return ok

        tpcds_unchanged: set[int] = set()

        def _run_candidate(num: int) -> bool:
            tid = id_by_number[num]
            if tid == LAZY_LOAD_TEST_ID and "tests/dialects/test_dialect.py" not in sel.patched_files:
                # The test only imports sqlglot in a subprocess. Phase 1 checked that import (plus
                # test_dialect collection import) with the patch applied; Phase 2 runs only when no
                # edited code runs at import (or the built import state is proven unchanged).
                return True if lazy_load_ok is None and task.phase == 2 else bool(lazy_load_ok)
            if task.tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
                suite = TPC_SUITE_BY_TEST_ID[tid]
                signal.alarm(120)
                try:
                    return verify_tpc_test(
                        suite,
                        task.tpc_mode,
                        query_mask=task.tpc_query_mask.get(num),
                        s2b_query_mask=task.tpc_s2b_query_mask.get(num),
                        patched_files=sel.patched_files,
                        edited_spans=sel.edited_spans,
                        unchanged_out=tpcds_unchanged if (suite == "tpcds" and task.tpc_mode == "fast_a") else None,
                    )
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if (
                tid == OPT_TPCDS_TEST_ID
                and select_tpcds_span(sel.patched_files, max_start=14, for_exec=False, edited_spans=sel.edited_spans)
                is not None
            ):
                signal.alarm(25)
                try:
                    cf_set = task.case_filter.get(num)
                    indices = {idx for _, idx in cf_set} if cf_set is not None else None
                    return verify_opt_tpcds(
                        sel.patched_files,
                        indices,
                        edited_spans=sel.edited_spans,
                        skip_indices=tpcds_unchanged or None,
                    )
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(sel):
                signal.alarm(15)
                try:
                    return verify_optimized_tpch(sel.patched_files)
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == MAX_NODES_TEST_ID and _can_fast_max_nodes(sel):
                signal.alarm(60)
                try:
                    return verify_max_nodes(
                        sel.patched_files,
                        position_free=tokenizer_edit_position_free(sel),
                        tok_untouched=task.tok_untouched,
                    )
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == GEN_MEM_TEST_ID:
                can_fast, is_scoped = _can_fast_gen_mem(sel)
                if can_fast and is_scoped:
                    signal.alarm(25)
                    try:
                        return verify_generator_memory(sel.patched_files, scoped_only=True)
                    except BaseException:
                        return False
                    finally:
                        signal.alarm(0)
            return _run_unittest(num)

        _t_last = [time.process_time()]

        def _mark(num: int, ok: bool) -> bool:
            """Record a verdict and stream it at once, so a timeout or crash keeps the passes so far."""
            if PROFILE_TESTS:
                now = time.process_time()
                sys.stderr.write("TESTCPU %.3f %s" % (now - _t_last[0], id_by_number[num]) + chr(10))
                _t_last[0] = now
            (passed if ok else failed).add(num)
            os.write(w_fd, f"{'P' if ok else 'F'} {num}\n".encode())
            if import_hung[0] or timeouts[0] >= 2:
                # The patched import hangs, or two tests ran out of time: the patch most likely
                # hangs, and every remaining test would burn its own timeout too. Stop; unverified
                # tests stay selected. (One timeout alone can be a test the patch merely slowed,
                # and stopping there would throw away cheap later proofs such as TPC-DS.)
                os.close(w_fd)
                os._exit(0)
            return ok

        # Step 1: If pushdown_predicates.py:36-37 lazy imports inflated `chosen`, verify clean import in ~10ms.
        if task.lazy_pp_nums:
            ok_lazy = False
            signal.alarm(10)
            try:
                ok_lazy = verify_lazy_athena_presto(sel.patched_files)
            except BaseException:
                ok_lazy = False
            finally:
                signal.alarm(0)
            if ok_lazy:
                for num in task.lazy_pp_nums:
                    _mark(num, True)

        verifiable = {
            n
            for n in (task.chosen - passed)
            if id_by_number[n] not in SUBPROCESS_TESTS
            and not (c_enabled and id_by_number[n] in PURE_PYTHON_ONLY_TESTS)
        }
        opt_tpcds_nums = [n for n in verifiable if id_by_number[n] == OPT_TPCDS_TEST_ID]
        decap_set = set(task.decap_targets) & verifiable

        def _propagate_fast_a_to_opt_tpcds() -> None:
            if not tpcds_unchanged or not opt_tpcds_nums:
                return
            opt_n = opt_tpcds_nums[0]
            if opt_n in passed or opt_n in failed:
                return
            fa_span = select_tpcds_span(sel.patched_files, max_start=14, for_exec=True, edited_spans=sel.edited_spans)
            opt_span = select_tpcds_span(sel.patched_files, max_start=14, for_exec=False, edited_spans=sel.edited_spans)
            if fa_span is None or opt_span is None or opt_span[1] is None:
                return
            opt_stop = opt_span[1]
            fa_stop = fa_span[1]
            can_skip = opt_stop == 0 or (
                opt_stop <= 7
                and fa_stop is not None
                and opt_stop <= fa_stop
                and all("leave_tables_isolated" not in (s or "") for s in sel.patched_files.values())
            )
            if not can_skip:
                return
            if opt_n in task.case_filter:
                all_q = {idx for _, idx in task.case_filter[opt_n]}
            else:
                meta = load_tpcds_ckpts_meta()
                q_ms = (meta.get("s2b_query_ms") or {}).get("tpcds") if meta else None
                if not q_ms:
                    return
                all_q = set(range(len(q_ms)))
            rem_q = all_q - tpcds_unchanged
            if not rem_q:
                _mark(opt_n, True)
            elif len(rem_q) < len(all_q):
                cpu_by_number[opt_n] = max(30, cpu_by_number[opt_n] * len(rem_q) // max(len(all_q), 1))

        if decap_set:
            total_decap_cpu = sum(
                _est_test_cpu_ms(
                    n,
                    id_by_number[n],
                    sel,
                    task.tpc_mode,
                    cpu_by_number,
                    task.tpc_query_mask,
                    task.tpc_s2b_query_mask,
                    c_exec_fast=task.c_exec_fast,
                    tok_untouched=task.tok_untouched,
                    leaf_compiled=leaf_compiled,
                )
                for n in decap_set
            )
            other_rem = verifiable - decap_set
            other_cpus = [cpu_by_number[n] for n in other_rem]
            will_verify_tail = (
                bool(other_rem)
                and max(other_cpus) <= MAX_SINGLE_TEST_MS
                and sum(other_cpus) <= MAX_COMMIT_VERIFY_MS
            )
            has_blockable_decap = False and any(
                id_by_number[n] in TPC_SUITE_BY_TEST_ID
                and TPC_SUITE_BY_TEST_ID[id_by_number[n]] == "tpcds"
                and task.tpc_mode != "fast_a"
                for n in decap_set
            )
            need_probe = has_blockable_decap or will_verify_tail
            probe_tests: list[int] = []
            if need_probe:
                other_ordered = sorted(
                    other_rem,
                    key=lambda n: (
                        0 if id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") in sys.modules else 1,
                        cpu_by_number[n],
                        ms_by_number[n],
                        id_by_number[n],
                    ),
                )
                probe_cpu = 0
                probe_mods: set[str] = set()
                for num in other_ordered:
                    c_ms = cpu_by_number[num]
                    m_name = id_by_number[num].split("::", 1)[0][:-3].replace("/", ".")
                    if c_ms <= 450 and probe_cpu + c_ms <= TPC_PROBE_BUDGET_MS:
                        if m_name not in probe_mods and len(probe_mods) >= 2:
                            continue
                        probe_mods.add(m_name)
                        probe_tests.append(num)
                        probe_cpu += c_ms
                        if len(probe_tests) >= 5:
                            break

            # Step 2a: Fail-fast probe on cheap unit tests.
            for num in probe_tests:
                if not _mark(num, _run_candidate(num)):
                    break

            # Step 2b: Makespan & worker-collapsing decapitation. Never block test_execute_tpch
            # (fast & high ROI even in full_cached) or fast_a TPC after a unit-test failure.
            def _tpc_blocked_after_fail(n: int) -> bool:
                return False

            decap_ordered = [n for n in task.decap_targets if n in decap_set]
            for idx, num in enumerate(decap_ordered):
                if num in passed or num in failed:
                    continue
                if _tpc_blocked_after_fail(num):
                    continue
                elig_rem = [x for x in decap_ordered[idx:] if x not in passed and x not in failed and not _tpc_blocked_after_fail(x)]
                if not elig_rem:
                    continue
                cur_pool = task.chosen - passed
                if not _has_decap_prefix_roi(
                    cur_pool,
                    elig_rem,
                    failed,
                    sel,
                    task.tpc_mode,
                    id_by_number,
                    ms_by_number,
                    cpu_by_number,
                    task.tpc_query_mask,
                    task.tpc_s2b_query_mask,
                    c_exec_fast=task.c_exec_fast,
                    tok_untouched=task.tok_untouched,
                    leaf_compiled=leaf_compiled,
                ):
                    continue
                ok_d = _mark(num, _run_candidate(num))
                if (
                    ok_d
                    and task.tpc_mode == "fast_a"
                    and TPC_SUITE_BY_TEST_ID.get(id_by_number[num]) == "tpcds"
                ):
                    _propagate_fast_a_to_opt_tpcds()

        # Step 2c: the tail, seeded by up to 4 tiny high-affinity probes when >=15 tests remain,
        # using Bayesian dynamic pass-rate ROI and continuing across multi-worker collapses.
        blocked_decap: set[int] = set()
        fix_narrow_only = sel.fixture_only and not VERIFY_FIXTURE_MAX_MS
        if not fix_narrow_only or (sel.fixture_changed_pairs and task.case_filter):
            per_mod_ms = 20 if task.phase == 1 else 4
            step2c_t0 = time.process_time()
            step2c_est_ms = 0
            s2c_failed = len(failed)
            s2c_passed = len(passed)

            rem_s2c_init = verifiable - passed - failed - blocked_decap
            if len(rem_s2c_init) >= 15 and not s2c_failed and not fix_narrow_only:
                edited_stems = {
                    Path(p).stem.lower()
                    for p in sel.patched_files
                    if p.endswith(".py")
                } - {"__init__", "core", "dialect", "optimizer", "parser", "generator", "expressions", "errors", "helper", "tokens"}
                if edited_stems:
                    affinity_cands = [
                        n
                        for n in rem_s2c_init
                        if cpu_by_number[n] <= 8
                        and id_by_number[n] not in (LAZY_LOAD_TEST_ID, OPT_TPCH_TEST_ID, MAX_NODES_TEST_ID, GEN_MEM_TEST_ID)
                        and id_by_number[n] not in TPC_SUITE_BY_TEST_ID
                        and not id_by_number[n].startswith("tests/test_dialect_imports.py::")
                        and any(st in id_by_number[n].lower() for st in edited_stems)
                    ]
                    affinity_cands.sort(
                        key=lambda n: (
                            0 if id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") in sys.modules else 1,
                            cpu_by_number[n],
                            ms_by_number[n],
                            id_by_number[n],
                        )
                    )
                    for num in affinity_cands[:4]:
                        step2c_est_ms += cpu_by_number[num]
                        if _mark(num, _run_candidate(num)):
                            s2c_passed += 1
                        else:
                            s2c_failed += 1
                            break

            for _ in range(STEP2C_ROUNDS):
                if step2c_est_ms > STEP2C_EST_CAP_MS or time.process_time() - step2c_t0 > STEP2C_CPU_S:
                    break
                n_obs = s2c_passed + s2c_failed
                if s2c_failed >= 3 and s2c_failed / max(n_obs, 1) > 0.45:
                    break
                if s2c_failed >= 2 and not s2c_passed and not STEP2C_KEEP_AFTER_FAIL:
                    break
                if s2c_failed == 0 and s2c_passed >= 2:
                    cur_roi = min(MIN_ROI_RATIO, 1.08)
                elif s2c_failed >= 2 and s2c_failed >= s2c_passed:
                    cur_roi = STEP2C_FAIL_ROI
                else:
                    cur_roi = MIN_ROI_RATIO
                s2c_cands = verifiable - passed - failed - blocked_decap
                if fix_narrow_only:
                    s2c_cands = {n for n in s2c_cands if n in task.case_filter and len(task.case_filter[n]) <= 8}
                cur_pool = task.chosen - passed
                batch = _best_drop_prefix(
                    cur_pool,
                    s2c_cands,
                    ms_by_number,
                    cpu_by_number,
                    MIN_GAIN_MS,
                    id_by_number=id_by_number,
                    per_mod_ms=per_mod_ms,
                    min_roi=cur_roi,
                )
                if not batch:
                    break
                step2c_est_ms += sum(cpu_by_number[n] for n in batch)
                batch_ok = True
                for num in batch:
                    ok_n = _mark(num, _run_candidate(num))
                    if ok_n:
                        s2c_passed += 1
                    else:
                        s2c_failed += 1
                        batch_ok = False
                        break
                if batch_ok:
                    _, cur_workers = split_with_cost([(ms_by_number[n], n) for n in (task.chosen - passed)])
                    if len(cur_workers) <= 1 and cur_roi <= min(MIN_ROI_RATIO, 1.08):
                        break
            # When the commit has 0 failures so far (>=2 passes), unlock dynamic 0-fail tail
            # clearing (full empty drops the 2.0s last-worker startup) and worker collapsing.
            if not failed and len(passed) >= 2 and time.process_time() - step2c_t0 <= 12.0:
                rem_tail = {
                    n
                    for n in (verifiable - passed)
                    if (
                        id_by_number[n] not in TPC_SUITE_BY_TEST_ID
                        or (task.tpc_mode is not None and TPC_SUITE_BY_TEST_ID[id_by_number[n]] == "tpch")
                    )
                    and id_by_number[n] != MAX_NODES_TEST_ID
                }
                if fix_narrow_only:
                    rem_tail = {n for n in rem_tail if n in task.case_filter and len(task.case_filter[n]) <= 8}
                cur_pool = task.chosen - passed
                if rem_tail:
                    cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
                    ran_full_clear = False
                    if cur_pool <= rem_tail:
                        rem_cpu = (
                            sum(cpu_by_number[n] for n in rem_tail)
                            + len({id_by_number[n].split("::", 1)[0] for n in rem_tail}) * per_mod_ms
                        )
                        if (
                            rem_cpu <= min(6500, int(cur_cost * 0.72))
                            and step2c_est_ms + rem_cpu <= 9000
                            and cur_cost >= rem_cpu * 1.15
                            and cur_cost - rem_cpu >= MIN_GAIN_MS
                        ):
                            ran_full_clear = True
                            tail_ordered = sorted(
                                rem_tail,
                                key=lambda n: (cpu_by_number[n], ms_by_number[n], id_by_number[n]),
                            )
                            for num in tail_ordered:
                                if not _mark(num, _run_candidate(num)):
                                    break
                    if not ran_full_clear and not failed:
                        batch = _best_drop_prefix(
                            cur_pool,
                            rem_tail,
                            ms_by_number,
                            cpu_by_number,
                            MIN_GAIN_MS,
                            id_by_number=id_by_number,
                            per_mod_ms=per_mod_ms,
                            min_roi=1.18,
                        )
                        if batch:
                            batch_cpu = (
                                sum(cpu_by_number[n] for n in batch)
                                + len({id_by_number[n].split("::", 1)[0] for n in batch}) * per_mod_ms
                            )
                            if (
                                batch_cpu <= min(4500, int(cur_cost * 0.65))
                                and step2c_est_ms + batch_cpu <= 8500
                            ):
                                tail_ordered = sorted(
                                    batch,
                                    key=lambda n: (cpu_by_number[n], ms_by_number[n], id_by_number[n]),
                                )
                                for num in tail_ordered:
                                    if not _mark(num, _run_candidate(num)):
                                        break
    except BaseException:
        pass
    finally:
        try:
            os.close(w_fd)
        except OSError:
            pass
        os._exit(0)


def _budget_scale(n_commits: int) -> float:
    """Budgets are sized for a 30-commit batch; the run's time limit grows with the batch
    (3 minutes in practice, 4.5 for the ~1.5x larger final)."""
    return max(1.0, n_commits / 30)


_CASE_DATA: tuple[dict, list[bytes] | None, dict[int, int]] | None = None


def _get_case_map() -> tuple[dict, list[bytes] | None, dict[int, int]]:
    global _CASE_DATA
    if _CASE_DATA is None:
        from blast_radius.precompile_cases import CASES_FILE

        if CASES_FILE.is_file():
            raw = marshal.loads(CASES_FILE.read_bytes())
            if isinstance(raw, dict) and raw.get("interned"):
                _CASE_DATA = (raw["cases"], raw["sets"], {})
            else:
                _CASE_DATA = (raw, None, {})
        else:
            _CASE_DATA = ({}, None, {})
    return _CASE_DATA


def _decode_mask(mask: int, number_by_index: list[int | None] | None) -> set[int]:
    out: set[int] = set()
    if not mask or number_by_index is None:
        return out
    m = mask
    while m:
        lsb = m & -m
        n = number_by_index[lsb.bit_length() - 1]
        if n is not None:
            out.add(n)
        m ^= lsb
    return out


def _case_plan(
    chosen: set[int],
    sel: Selection,
    id_by_number: dict[int, str],
    base_cpu: dict[int, int],
    number_by_index: list[int | None] | None = None,
    opt_c_scale: float = 1.0,
) -> tuple[
    dict[int, set[tuple[str, int]]],
    dict[int, tuple[int, int]],
    dict[int, int],
    dict[int, tuple[int, int]],
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    """Compute per-case/subTest/block/Validator/TPC query filters and scaled CPU estimates for `chosen`."""
    case_filter: dict[int, set[tuple[str, int]]] = {}
    subtest_filter: dict[int, tuple[int, int]] = {}
    block_filter: dict[int, int] = {}
    validator_filter: dict[int, tuple[int, int]] = {}
    tpc_query_mask: dict[int, int] = {}
    tpc_s2b_query_mask: dict[int, int] = {}
    cpu_override: dict[int, int] = {}

    non_sg_nums = _decode_mask(sel.non_sqlglot_mask, number_by_index)
    fix_mapped_nums = _decode_mask(sel.fixture_mapped_mask, number_by_index)
    sg_nums = _decode_mask(sel.sqlglot_mask, number_by_index)

    if sel.edited_spans is not None:
        cases_dict, sets_raw, sets_cache = _get_case_map()

        def _decode_idx(idx: int) -> int:
            if sets_raw is None:
                return idx
            val = sets_cache.get(idx)
            if val is None:
                val = int.from_bytes(sets_raw[idx], "little")
                sets_cache[idx] = val
            return val

        for num in chosen:
            if num in non_sg_nums:
                continue
            tid = id_by_number[num]
            entry = cases_dict.get(tid)
            if entry is None:
                continue
            kind = entry.get("kind", "check_file")
            if num in fix_mapped_nums and kind != "check_file":
                continue
            mask = 0
            for path, first, last in sel.edited_spans:
                per_line = entry["lines"].get(path)
                if per_line:
                    for line in range(first, last + 1):
                        val = per_line.get(line)
                        if val is not None:
                            mask |= _decode_idx(val)
            if kind == "check_file" and num in fix_mapped_nums:
                if num in sg_nums and mask == 0:
                    continue
                cases = entry["cases"]
                sg_matched = {tuple(cases[k]) for k in range(len(cases)) if mask >> k & 1} if num in sg_nums else set()
                entry_fks = {fk for fk, _ in cases}
                changed_pairs = sel.fixture_changed_pairs or {}
                fix_matched = {
                    (fk, idx) for fk in entry_fks if fk in changed_pairs for idx in changed_pairs[fk]
                }
                union = sg_matched | fix_matched
                if len(union) < len(cases) and (
                    union or (num not in sg_nums and any(fk in changed_pairs for fk in entry_fks))
                ):
                    case_filter[num] = union
                    cpu_override[num] = max(25, base_cpu[num] * (len(union) + 1) // (len(cases) + 1) + 15)
                continue
            if kind == "tpc_stage1":
                if mask:
                    tpc_query_mask[num] = mask
                else:
                    # Only skip Stage 1 (mask=0) if Stage 2a actually executed an edited span.
                    s2a = entry.get("s2a_lines", {})
                    if any(
                        first <= ln <= last
                        for path, first, last in sel.edited_spans
                        for ln in s2a.get(path, ())
                    ):
                        tpc_query_mask[num] = 0
                # Also compute per-query Stage-2b mask for fast_b / full_cached.
                s2b_lines = entry.get("s2b_lines")
                if s2b_lines:
                    s2b_mask = 0
                    for path, first, last in sel.edited_spans:
                        per_line = s2b_lines.get(path)
                        if per_line:
                            for line in range(first, last + 1):
                                val = per_line.get(line)
                                if val is not None:
                                    s2b_mask |= _decode_idx(val)
                    if s2b_mask:
                        tpc_s2b_query_mask[num] = s2b_mask
                continue
            # Soundness guard: if a selected test matched 0 recorded cases, run it unfiltered.
            if not mask:
                continue
            if kind == "check_file":
                cases = entry["cases"]
                matched = {tuple(cases[k]) for k in range(len(cases)) if mask >> k & 1}
                if len(matched) < len(cases):
                    case_filter[num] = matched
                    cpu_override[num] = max(40, base_cpu[num] * (len(matched) + 1) // (len(cases) + 1) + 20)
            elif kind == "subtest":
                count = entry["count"]
                hits = mask.bit_count()
                if hits < count:
                    subtest_filter[num] = (mask, entry.get("stride", 1))
                    cpu_override[num] = max(40, base_cpu[num] * (hits + 1) // (count + 1) + 20)
            elif kind == "blocks":
                count = entry["count"]
                hits = mask.bit_count()
                if hits < count:
                    block_filter[num] = mask
                    b_ms = entry.get("block_ms")
                    if b_ms and len(b_ms) == count:
                        tot_b = max(sum(b_ms), 1)
                        hit_b = sum(b_ms[i] for i in range(count) if mask >> i & 1)
                        cpu_override[num] = max(25, base_cpu[num] * hit_b // tot_b + 15)
                    else:
                        cpu_override[num] = max(25, base_cpu[num] * (hits + 1) // (count + 1) + 15)
            elif kind == "validator":
                count = entry["count"]
                full_mask = mask | entry.get("always", 0)
                hits = full_mask.bit_count()
                if hits < count:
                    validator_filter[num] = (full_mask, entry.get("stride", 1))
                    cpu_override[num] = max(22, base_cpu[num] * (hits + 1) // (count + 1) + 10)

    # Narrow fixture-only optimizer commits with <= 8 changed pairs to exact changed check_file cases.
    if sel.fixture_only and sel.fixture_changed_pairs:
        all_pairs_mapped = bool(sel.patched_files) and all(
            p.startswith("tests/fixtures/") and p[len("tests/fixtures/") :] in sel.fixture_changed_pairs
            for p in sel.patched_files
        )
        if all_pairs_mapped:
            cases_dict, _, _ = _get_case_map()
            for num in chosen:
                entry = cases_dict.get(id_by_number[num])
                if entry is None or entry.get("kind") != "check_file":
                    continue
                cases = entry["cases"]
                entry_fks = {fk for fk, _ in cases}
                has_fk = any(fk in sel.fixture_changed_pairs for fk in entry_fks)
                matched = {
                    (fk, idx)
                    for fk in entry_fks
                    if fk in sel.fixture_changed_pairs
                    for idx in sel.fixture_changed_pairs[fk]
                }
                if has_fk and len(matched) < len(cases):
                    case_filter[num] = matched
                    cpu_override[num] = max(25, base_cpu[num] * len(matched) // max(len(cases), 1) + 15)

    # Apply TPC-DS optimizer rule-stage checkpoint/suffix-span CPU estimate to TestOptimizer.test_tpcds.
    span_info = select_tpcds_span(sel.patched_files, max_start=14, for_exec=False, edited_spans=sel.edited_spans)
    if span_info is not None:
        _, _, ckpt_est_ms = span_info
        for num in chosen:
            if id_by_number[num] == OPT_TPCDS_TEST_ID:
                if num in case_filter:
                    scaled = max(35, round(ckpt_est_ms * opt_c_scale * (len(case_filter[num]) + 1) / 100))
                else:
                    scaled = max(40, round(ckpt_est_ms * opt_c_scale))
                cpu_override[num] = min(cpu_override.get(num, base_cpu[num]), scaled)

    return case_filter, subtest_filter, block_filter, validator_filter, tpc_query_mask, tpc_s2b_query_mask, cpu_override


def plan_verification(
    commits: list[tuple[str, set[int], Selection]],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    base_cpu: dict[int, int],
    number_by_index: list[int | None] | None = None,
    base_cpu_c: dict[int, int] | None = None,
    tpc_c_ms: dict[str, int] | None = None,
) -> tuple[list[VerifyTask], list[VerifyTask]]:
    """Select and prioritize commits worth runtime verification or makespan decapitation."""
    manifest = load_c_manifest() if ENABLE_MYPYC else None
    all_c_groups = frozenset(manifest.get("groups", ())) if manifest else frozenset()
    candidates: list[VerifyTask] = []
    for commit, chosen, sel in commits:
        if not chosen or not sel.patched_files or sel.reason.startswith("everything"):
            continue
        if manifest is not None and ENABLE_MYPYC:
            if "sqlglot/__init__.py" in sel.patched_files or any(
                (p.startswith("sqlglot/") and not p.endswith(".py")) or sel.patched_files[p] is None
                for p in sel.patched_files
            ):
                disabled_c = all_c_groups
            else:
                disabled_c = disabled_groups_for(sel.patched_files)
            c_enabled = bool(all_c_groups - disabled_c)
        else:
            disabled_c = frozenset()
            c_enabled = False
        leaf_compiled = manifest is not None and ("leaf" not in disabled_c)
        c_exec_fast = bool(
            manifest is not None
            and ENABLE_MYPYC
            and not (disabled_c & {"leaf", "expr", "parse", "gen", "exec_table"})
        )
        c_bucket = phase2_c_bucket(disabled_c) if c_enabled else "fallback_c"
        tok_untouched = not tokenizer_edit_touches_max_nodes(sel)

        if base_cpu_c is None or not c_enabled or bool(disabled_c & {"leaf", "expr"}):
            commit_base_cpu = dict(base_cpu)
        elif disabled_c & {"parse", "gen"}:
            commit_base_cpu = {n: round(0.45 * base_cpu_c[n] + 0.55 * base_cpu[n]) for n in chosen}
        elif disabled_c:
            commit_base_cpu = {
                n: (
                    round(0.40 * base_cpu_c[n] + 0.60 * base_cpu[n])
                    if ms_by_number[n] >= 300 and id_by_number[n].startswith("tests/test_optimizer.py::")
                    else base_cpu_c[n]
                )
                for n in chosen
            }
        else:
            commit_base_cpu = {n: base_cpu_c[n] for n in chosen}

        opt_py_ms = next((base_cpu[n] for n in chosen if id_by_number[n] == OPT_TPCDS_TEST_ID), 0)
        if c_exec_fast:
            opt_c_scale = (
                min(1.0, max(0.22, tpc_c_ms["opt_tpcds_full"] / max(opt_py_ms, 1)))
                if (tpc_c_ms and tpc_c_ms.get("opt_tpcds_full") and opt_py_ms > 0)
                else 0.40
            )
        else:
            opt_c_scale = 1.0

        touches_test_py = any(p.startswith("tests/") and p.endswith(".py") for p in sel.patched_files)
        (
            case_filter,
            subtest_filter,
            block_filter,
            validator_filter,
            tpc_query_mask,
            tpc_s2b_query_mask,
            cpu_override,
        ) = _case_plan(
            chosen,
            sel,
            id_by_number,
            commit_base_cpu,
            number_by_index=number_by_index,
            opt_c_scale=opt_c_scale,
        )
        cpu_by_number = {n: cpu_override.get(n, commit_base_cpu[n]) for n in chosen}
        for n in chosen:
            if cpu_by_number[n] != base_cpu[n]:
                cpu_override[n] = cpu_by_number[n]

        tids = [id_by_number[n] for n in chosen]
        verifiable = {
            n
            for n in chosen
            if id_by_number[n] not in SUBPROCESS_TESTS
            and not (c_enabled and id_by_number[n] in PURE_PYTHON_ONLY_TESTS)
        }
        can_fast_gm, is_scoped_gm = _can_fast_gen_mem(sel)
        gen_mem_num = next(
            (
                n
                for n in chosen
                if id_by_number[n] == GEN_MEM_TEST_ID
                and c_enabled
                and can_fast_gm
                and not touches_test_py
            ),
            None,
        )
        if not verifiable and gen_mem_num is None:
            continue

        static_cost_ms, _ = split_with_cost([(ms_by_number[n], n) for n in chosen])
        # test_lazy_load only checks `import sqlglot`: safe in Phase 2 when zero import-time code changed.
        lazy_safe_p2 = not sel.import_checks and "tests/dialects/test_dialect.py" not in sel.patched_files
        can_phase2 = (
            ENABLE_HOTPATCH
            and sel.hot_patch_targets is not None
            and not sel.lazy_pp_mask
            and not touches_test_py
            and not any(
                tid in FRESH_IMPORT_TESTS and not (lazy_safe_p2 and tid == LAZY_LOAD_TEST_ID)
                for tid in tids
            )
        )
        phase = 2 if can_phase2 else 1
        overhead_ms = 35 if phase == 2 else 320
        per_mod_ms = 4 if phase == 2 else 20

        lazy_pp_nums: set[int] = set()
        if sel.lazy_pp_mask and number_by_index is not None:
            x = sel.lazy_pp_mask
            while x:
                lsb = x & -x
                idx = lsb.bit_length() - 1
                num = number_by_index[idx]
                if num is not None and num in verifiable:
                    lazy_pp_nums.add(num)
                x ^= lsb

        tpc_nums = [n for n in (verifiable - lazy_pp_nums) if id_by_number[n] in TPC_SUITE_BY_TEST_ID]
        tpc_mode: str | None = None
        if tpc_nums and ENABLE_TPC_FAST:
            tpc_mode = select_tpc_mode(sel)
            if tpc_mode == "full_cached" and not ENABLE_TPC_FAST_B:
                tpc_mode = None
            elif tpc_mode == "fast_b" and not ENABLE_TPC_FAST_B:
                if any(
                    _est_test_cpu_ms(
                        n,
                        id_by_number[n],
                        sel,
                        tpc_mode,
                        cpu_by_number,
                        tpc_query_mask,
                        tpc_s2b_query_mask,
                        c_exec_fast=c_exec_fast,
                        tok_untouched=tok_untouched,
                        leaf_compiled=leaf_compiled,
                        tpc_c_ms=tpc_c_ms,
                    )
                    > FAST_B_MAX_EST_MS
                    for n in tpc_nums
                ):
                    tpc_mode = None

        for num in verifiable:
            sp = _special_cpu_ms(
                num,
                id_by_number[num],
                sel,
                tpc_mode,
                tpc_query_mask,
                tpc_s2b_query_mask,
                c_exec_fast=c_exec_fast,
                tok_untouched=tok_untouched,
                leaf_compiled=leaf_compiled,
                tpc_c_ms=tpc_c_ms,
            )
            if sp is not None:
                scaled_sp = round(sp * SPECIAL_CPU_SCALE)
                if scaled_sp < cpu_by_number[num]:
                    cpu_override[num] = scaled_sp
                    cpu_by_number[num] = scaled_sp

        rem_verifiable = verifiable - lazy_pp_nums

        # Identify high-ROI heavy and medium peak-shaving targets inside rem_verifiable (plus pure-Py gen_mem).
        raw_decap: list[tuple[int, int, int]] = []  # (sim_ms, est_cpu_ms, num)
        if gen_mem_num is not None and gen_mem_num not in lazy_pp_nums:
            gm_cpu = (450 if is_scoped_gm else 550) + 150
            if gm_cpu < ms_by_number[gen_mem_num]:
                raw_decap.append((ms_by_number[gen_mem_num], gm_cpu, gen_mem_num))
        for num in rem_verifiable:
            if sel.fixture_only and not VERIFY_FIXTURE_MAX_MS and not (num in case_filter and len(case_filter[num]) <= 8):
                continue
            tid = id_by_number[num]
            sim_ms = ms_by_number[num]
            if tid in TPC_SUITE_BY_TEST_ID:
                if tpc_mode is None:
                    continue
                c_ms = _est_test_cpu_ms(
                    num,
                    tid,
                    sel,
                    tpc_mode,
                    cpu_by_number,
                    tpc_query_mask,
                    tpc_s2b_query_mask,
                    c_exec_fast=c_exec_fast,
                    tok_untouched=tok_untouched,
                    leaf_compiled=leaf_compiled,
                    tpc_c_ms=tpc_c_ms,
                )
                raw_decap.append((sim_ms, c_ms, num))
            elif tid in (OPT_TPCH_TEST_ID, MAX_NODES_TEST_ID, GEN_MEM_TEST_ID, LAZY_LOAD_TEST_ID):
                c_ms = _est_test_cpu_ms(
                    num,
                    tid,
                    sel,
                    tpc_mode,
                    cpu_by_number,
                    tpc_query_mask,
                    tpc_s2b_query_mask,
                    c_exec_fast=c_exec_fast,
                    tok_untouched=tok_untouched,
                    leaf_compiled=leaf_compiled,
                    tpc_c_ms=tpc_c_ms,
                )
                if c_ms < sim_ms:
                    raw_decap.append((sim_ms, c_ms, num))
            elif sim_ms >= 150:
                c_ms = cpu_by_number[num]
                if c_ms <= sim_ms * DECAP_CPU_RATIO:
                    raw_decap.append((sim_ms, c_ms, num))

        raw_decap = sorted(raw_decap, key=lambda item: (item[1] - item[0], item[2]))[:30]
        cur_pool = chosen - lazy_pp_nums
        cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
        base_net = (static_cost_ms - cur_cost) - (15 if lazy_pp_nums else 0)
        best_net = base_net
        best_prefix: list[int] = []
        orders = (
            sorted(raw_decap, key=lambda item: (-item[0], item[1], item[2])),
            sorted(raw_decap, key=lambda item: (item[1] / item[0], -item[0], item[2])),
        )
        for order in orders:
            order_best, order_prefix = base_net, []
            chosen_decap: list[int] = []
            acc_decap_cpu = 0
            trial_pool = set(cur_pool)
            for _sim_ms, c_ms, num in order:
                trial_pool.discard(num)
                acc_decap_cpu += c_ms
                new_cost, _ = split_with_cost([(ms_by_number[n], n) for n in trial_pool])
                net = (static_cost_ms - new_cost) - acc_decap_cpu * CPU_WEIGHT - (15 if lazy_pp_nums else 0)
                chosen_decap.append(num)
                if net >= order_best + 200:
                    order_best, order_prefix = net, list(chosen_decap)
            if order_best > best_net:
                best_net, best_prefix = order_best, order_prefix

        pure_py_decap = [n for n in best_prefix if n == gen_mem_num]
        decap_targets = [n for n in best_prefix if n != gen_mem_num]
        decap_set = set(decap_targets)
        all_decap_set = set(best_prefix)

        def plan_tail(pool: set[int], cands: set[int]) -> tuple[list[int], int]:
            if sel.fixture_only and not VERIFY_FIXTURE_MAX_MS:
                if not (sel.fixture_changed_pairs and case_filter):
                    return [], 0  # a fixture edit almost always breaks the tests reading it
                cands = {n for n in cands if n in case_filter and len(case_filter[n]) <= 8}
                if not cands:
                    return [], 0
            prefix = _best_drop_prefix(
                pool,
                cands,
                ms_by_number,
                cpu_by_number,
                MIN_GAIN_MS,
                id_by_number=id_by_number,
                per_mod_ms=per_mod_ms,
                min_roi=MIN_ROI_RATIO,
            )
            mod_count = len({id_by_number[n].split("::", 1)[0] for n in prefix})
            return prefix, sum(cpu_by_number[n] for n in prefix) + mod_count * per_mod_ms

        decap_cpu = sum(
            _est_test_cpu_ms(
                n,
                id_by_number[n],
                sel,
                tpc_mode,
                cpu_by_number,
                tpc_query_mask,
                tpc_s2b_query_mask,
                c_exec_fast=c_exec_fast,
                tok_untouched=tok_untouched,
                leaf_compiled=leaf_compiled,
                tpc_c_ms=tpc_c_ms,
            )
            for n in decap_targets
        ) + (15 if lazy_pp_nums else 0) + (len(pure_py_decap) * ((450 if is_scoped_gm else 550) + 150))
        pool_after_decap = chosen - lazy_pp_nums - all_decap_set
        tail_prefix, tail_cpu = plan_tail(pool_after_decap, rem_verifiable - decap_set)
        cost_after, _ = split_with_cost([(ms_by_number[n], n) for n in pool_after_decap - set(tail_prefix)])
        own_overhead = overhead_ms if (decap_targets or tail_prefix or lazy_pp_nums) else 0
        est_cpu_ms = own_overhead + decap_cpu + tail_cpu
        saved_ms = static_cost_ms - cost_after
        expected_gain_ms = saved_ms - est_cpu_ms * CPU_WEIGHT
        has_heavy_tpc_or_max = any(
            id_by_number[n] in TPC_SUITE_BY_TEST_ID or id_by_number[n] == MAX_NODES_TEST_ID for n in decap_targets
        ) or bool(pure_py_decap)
        min_req = TPC_MIN_GAIN_MS if has_heavy_tpc_or_max else MIN_DECAP_GAIN_MS if decap_targets else MIN_GAIN_MS
        roi_ok = (saved_ms >= est_cpu_ms * MIN_ROI_RATIO * CPU_WEIGHT) or (saved_ms >= 5000 and expected_gain_ms >= 2500)
        if (expected_gain_ms < min_req or not roi_ok) and best_prefix:
            # Without the decapitation targets, the tail alone may still pay off.
            pure_py_decap = []
            decap_targets = []
            tail_prefix, tail_cpu = plan_tail(chosen - lazy_pp_nums, rem_verifiable)
            cost_after, _ = split_with_cost([(ms_by_number[n], n) for n in chosen - lazy_pp_nums - set(tail_prefix)])
            est_cpu_ms = overhead_ms + (15 if lazy_pp_nums else 0) + tail_cpu
            saved_ms = static_cost_ms - cost_after
            expected_gain_ms = saved_ms - est_cpu_ms * CPU_WEIGHT
            min_req = MIN_GAIN_MS
            roi_ok = saved_ms >= est_cpu_ms * MIN_ROI_RATIO * CPU_WEIGHT
        if not (decap_targets or pure_py_decap or tail_prefix or lazy_pp_nums) or expected_gain_ms < min_req or not roi_ok:
            continue

        candidates.append(
            VerifyTask(
                commit=commit,
                chosen=chosen,
                selection=sel,
                phase=phase,
                static_cost_ms=static_cost_ms,
                est_cpu_ms=est_cpu_ms,
                expected_gain_ms=expected_gain_ms,
                tpc_mode=tpc_mode,
                lazy_pp_nums=lazy_pp_nums,
                decap_targets=decap_targets,
                case_filter=case_filter,
                subtest_filter=subtest_filter,
                block_filter=block_filter,
                validator_filter=validator_filter,
                tpc_query_mask=tpc_query_mask,
                tpc_s2b_query_mask=tpc_s2b_query_mask,
                cpu_override=cpu_override,
                disabled_c_groups=disabled_c,
                c_bucket=c_bucket,
                c_exec_fast=c_exec_fast,
                pure_py_decap=pure_py_decap,
                tok_untouched=tok_untouched,
            )
        )

    # Deterministically prioritize highest ROI (net savings / CPU) within total budget.
    candidates.sort(
        key=lambda t: (-t.expected_gain_ms / max(t.est_cpu_ms, 1), t.commit)
    )
    budget_used = 0
    budget = TOTAL_VERIFY_BUDGET_MS * _budget_scale(len(commits))
    phase1: list[VerifyTask] = []
    phase2: list[VerifyTask] = []
    for task in candidates:
        if budget_used + task.est_cpu_ms > budget:
            continue
        budget_used += task.est_cpu_ms
        if task.phase == 1:
            phase1.append(task)
        else:
            phase2.append(task)
    phase1.sort(key=lambda t: t.commit)
    phase2.sort(key=lambda t: t.commit)
    # Free the case map before forking children (tasks already hold extracted filters).
    global _CASE_DATA
    _CASE_DATA = None
    return phase1, phase2


def _prewarm_for_phase2(
    phase2_tasks: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
) -> None:
    """Import sqlglot core and shared/first-run test modules once in parent before Phase-2 forks."""
    if not phase2_tasks:
        return
    os.chdir(SOURCE)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    _patch_pool_references()
    import sqlglot  # noqa: F401

    mod_task_count: dict[str, int] = {}
    needed_mods: set[str] = set()
    for task in phase2_tasks:
        for rel_path, _ in task.selection.hot_patch_targets or ():
            needed_mods.add(module_name(rel_path))
        for rel_path, _, _, _ in task.selection.attr_hot_targets or ():
            needed_mods.add(module_name(rel_path))
        for rel_path, *_ in task.selection.dict_hot_targets or ():
            needed_mods.add(module_name(rel_path))
        verifiable = [n for n in task.chosen if id_by_number[n] not in SUBPROCESS_TESTS]
        if not verifiable:
            continue
        ordered = sorted(verifiable, key=lambda n: (cpu_by_number[n], ms_by_number[n], id_by_number[n]))
        first_file = id_by_number[ordered[0]].split("::", 1)[0]
        first_mod = first_file[:-3].replace("/", ".")
        if PREWARM_SINGLE and first_mod != "tests.test_executor":
            needed_mods.add(first_mod)
        for num in task.decap_targets:
            dmod = id_by_number[num].split("::", 1)[0][:-3].replace("/", ".")
            if PREWARM_SINGLE and dmod != "tests.test_executor":
                needed_mods.add(dmod)
        task_mods = {id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") for n in verifiable}
        for mod in task_mods:
            mod_task_count[mod] = mod_task_count.get(mod, 0) + 1

    for mod, count in mod_task_count.items():
        if count >= 2:
            needed_mods.add(mod)

    # Install lazy numpy/pandas stubs unconditionally in Phase-2 parent unless a task needs real
    # TPC pandas/numpy or patches test_executor.py — saves ~1.5s in any child importing test_executor.
    lazy_data_ok = not any(
        "tests/test_executor.py" in t.selection.patched_files
        or any(
            _real_tpc_possible(t.tpc_mode, t.selection, n, id_by_number, ms_by_number, cpu_by_number)
            for n in t.chosen
        )
        for t in phase2_tasks
    )
    if lazy_data_ok:
        _install_lazy_data_libs()

    for mod_name in sorted(needed_mods):
        if mod_name not in sys.modules:
            try:
                mod = importlib.import_module(mod_name)
                wrap_duckdb_if_loaded()
                if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                    _patch_pool_references()
                if mod_name == "tests.test_optimizer":
                    _install_case_filter(mod)
                    _install_lazy_optimizer_setup(mod.TestOptimizer)
                elif mod_name == "tests.test_executor" and lazy_data_ok:
                    _install_lazy_executor_setup(mod.TestExecutor)
            except Exception as err:
                log(f"prewarm skip {mod_name}: {err}")

    # Pre-install subTest AST wrappers and Validator filter once in parent for Phase-2 COW sharing.
    has_val_filter = any(t.validator_filter for t in phase2_tasks)
    if has_val_filter and "tests.dialects.test_dialect" in sys.modules:
        _install_validator_filter()
    for task in phase2_tasks:
        for num in task.subtest_filter:
            parts = id_by_number[num].split("::")
            if len(parts) == 3:
                mod_name = parts[0][:-3].replace("/", ".")
                mod = sys.modules.get(mod_name)
                if mod is not None and hasattr(mod, parts[1]):
                    _install_subtest_filter(mod, getattr(mod, parts[1]), parts[2])
        for num in task.block_filter:
            parts = id_by_number[num].split("::")
            if len(parts) == 3:
                mod_name = parts[0][:-3].replace("/", ".")
                mod = sys.modules.get(mod_name)
                if mod is not None and hasattr(mod, parts[1]):
                    _install_block_filter(mod, getattr(mod, parts[1]), parts[2])


def _real_tpc_possible(task_tpc_mode, sel, n: int, id_by_number, ms_by_number, cpu_by_number) -> bool:
    """True iff the TPC test `n` could be run for real (real numpy/pandas/DuckDB data) in a child.

    A test the planner prices above its own simulated time can never be a profitable
    verification target, so it is never run: its presence in `chosen` needs no real data libs.
    (If it ever ran under lazy stubs it would fail, i.e. stay selected: never a miss.)"""
    tid = id_by_number[n]
    needs = (tid in TPC_SUITE_BY_TEST_ID and not task_tpc_mode) or (
        tid == OPT_TPCH_TEST_ID and not _can_fast_opt_tpch(sel)
    )
    if not needs:
        return False
    return cpu_by_number[n] * MIN_ROI_RATIO * CPU_WEIGHT < ms_by_number[n]


def _kill_group(pid: int) -> None:
    """SIGKILL the process group led by `pid` (the verification child and its subprocesses)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _run_batch_parallel(
    tasks: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    out_fd: int | None,
    stop_at: float,
    verified: dict[str, set[int]],
    started: float,
) -> None:
    """Run a batch of independent forked verification tasks with up to MAX_PARALLEL_CHILDREN concurrency."""
    queue = list(tasks)
    # active: r_fd -> (pid, task, deadline, chunks_list)
    active: dict[int, tuple[int, VerifyTask, float, list[bytes]]] = {}

    reaped: dict[int, tuple[int, object]] = {}

    def finish_child(r_fd: int, timed_out: bool) -> None:
        pid, task, limit_deadline, chunks = active.pop(r_fd)
        # Kill the whole group: on a timeout the child itself, and in every case any subprocess
        # it left behind (which would otherwise keep running and hold the pipe open).
        _kill_group(pid)
        # Drain what the child streamed before it exited or was killed: each "P n" line is a
        # test that ran to completion and passed, so it stays valid after a kill or crash.
        try:
            while chunk := os.read(r_fd, 65536):
                chunks.append(chunk)
        except OSError:
            pass
        os.close(r_fd)
        if pid in reaped:
            status, rusage = reaped.pop(pid)
        else:
            _, status, rusage = os.wait4(pid, 0)
        child_cpu = rusage.ru_utime + rusage.ru_stime
        raw = b"".join(chunks)
        if raw.startswith(b"FAIL_PATCH"):
            log(f"{task.commit}: verify fallback p{task.phase} ({child_cpu:.3f}s, patch not applied)")
            verified[task.commit] = task.chosen
            return
        lines = raw.split(b"\n")
        passed_set = {int(line[2:]) for line in lines if line.startswith(b"P ")}
        failed_ids = [id_by_number[int(line[2:])].split("::")[-1] for line in lines if line.startswith(b"F ")]
        mtag = f" tpc={task.tpc_mode}" if task.tpc_mode else ""
        dtag = f" decap={len(task.decap_targets)}" if task.decap_targets else ""
        end = " TIMEOUT" if timed_out else ("" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f" status={status}")
        log(
            f"{task.commit}: p{task.phase}{mtag}{dtag} cpu={child_cpu:.3f}s est={task.est_cpu_ms / 1000:.3f}s gain={task.expected_gain_ms / 1000:.1f}s "
            f"pass={len(passed_set)} fail={len(failed_ids)}/{len(task.chosen)}{end} failed={failed_ids[:5]}"
        )
        verified[task.commit] = task.chosen - passed_set

    while queue or active:
        # Run 1 child at a time initially to avoid SMT cache contention (+0.7s CPU);
        # automatically scale to 2-core parallelism if wall-clock elapsed exceeds 35s.
        cur_limit = max(1, MAX_PARALLEL_CHILDREN) if time.monotonic() - started > 35.0 else 1
        while queue and len(active) < cur_limit:
            now = time.monotonic()
            remaining = stop_at - now
            if remaining <= 0:
                for skipped in queue:
                    log(f"{skipped.commit}: verify skipped, wall budget spent")
                queue.clear()
                break
            task = queue.pop(0)
            limit_cpu_ms = task.est_cpu_ms
            if task.tpc_mode == "fast_a" and any(id_by_number[n] in TPC_SUITE_BY_TEST_ID for n in task.decap_targets):
                # fast_a re-executes the queries whose plans changed, up to a full_cached run.
                limit_cpu_ms += round((TPC_REEXEC_BUDGET_S + TPC_DUCK_BUDGET_S) * 2000 / CHILD_LIMIT_FACTOR)
            limit_s = min(CHILD_LIMIT_BASE_S + CHILD_LIMIT_FACTOR * limit_cpu_ms / 1000, remaining)
            pid, r_fd = _start_child_verification(task, id_by_number, ms_by_number, cpu_by_number, out_fd)
            active[r_fd] = (pid, task, time.monotonic() + limit_s, [])

        if not active:
            break

        now = time.monotonic()
        expired = [r_fd for r_fd, (_, _, dl, _) in active.items() if now >= dl]
        for r_fd in expired:
            finish_child(r_fd, timed_out=True)
        if not active:
            continue

        # A child that exited while a subprocess of its still holds the pipe never reaches EOF:
        # notice the exit itself and finish it now.
        # waitpid(-1) also reaps killed subprocesses re-parented to us (we may be PID 1), which
        # would otherwise pile up as zombies against the container's process limit.
        fd_by_pid = {pid: r_fd for r_fd, (pid, _, _, _) in active.items()}
        while True:
            try:
                got_pid, status, rusage = os.wait4(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if got_pid == 0:
                break
            if got_pid in fd_by_pid:
                reaped[got_pid] = (status, rusage)
                finish_child(fd_by_pid.pop(got_pid), timed_out=False)
        if not active:
            continue

        min_rem = min(dl - time.monotonic() for _, _, dl, _ in active.values())
        if min_rem <= 0:
            continue
        ready, _, _ = select.select(list(active.keys()), [], [], min(min_rem, 0.25))
        for r_fd in ready:
            if r_fd not in active:
                continue
            try:
                chunk = os.read(r_fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                finish_child(r_fd, timed_out=False)
            else:
                active[r_fd][3].append(chunk)


def _run_pure_py_decap(
    tasks: list[VerifyTask],
    stop_at: float,
) -> dict[str, set[int]]:
    """Run pure-Python generator_memory verifications before any C extensions load."""
    passed: dict[str, set[int]] = {}
    for task in tasks:
        if not task.pure_py_decap or time.monotonic() >= stop_at:
            continue
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r_fd)
            try:
                os.setpgid(0, 0)
            except OSError:
                pass
            ok = False
            try:
                os.chdir(SOURCE)
                if str(SOURCE) not in sys.path:
                    sys.path.insert(0, str(SOURCE))
                sys.meta_path.insert(0, _OverlayFinder(task.selection.patched_files, root=SOURCE))
                _install_open_overlay(task.selection.patched_files)
                signal.signal(signal.SIGALRM, _raise_timeout)
                signal.alarm(20)
                _, is_scoped = _can_fast_gen_mem(task.selection)
                ok = verify_generator_memory(task.selection.patched_files, scoped_only=is_scoped)
            except BaseException:
                ok = False
            finally:
                signal.alarm(0)
                try:
                    os.write(w_fd, b"1" if ok else b"0")
                except OSError:
                    pass
                os.close(w_fd)
                os._exit(0)
        os.close(w_fd)
        chunks = []
        rem = max(0.1, stop_at - time.monotonic())
        ready, _, _ = select.select([r_fd], [], [], min(rem, 25.0))
        if ready:
            try:
                while chunk := os.read(r_fd, 64):
                    chunks.append(chunk)
            except OSError:
                pass
        _kill_group(pid)
        os.close(r_fd)
        os.waitpid(pid, 0)
        if b"".join(chunks).startswith(b"1"):
            passed[task.commit] = set(task.pure_py_decap)
            log(f"{task.commit}: pure_py_decap verified {len(task.pure_py_decap)} test(s)")
    return passed


def verify_batches(
    phase1: list[VerifyTask],
    phase2: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    out_fd: int | None,
    cpu_by_number: dict[int, int] | None = None,
    n_commits: int = 30,
) -> dict[str, set[int]]:
    """Run pure-Py decap, Phase-1 (clean fork + modular C), then bucketed Phase-2 coordinators."""
    verified: dict[str, set[int]] = {}
    if not hasattr(os, "fork"):
        return verified
    if cpu_by_number is None:
        cpu_by_number = ms_by_number

    all_tasks = phase1 + phase2
    if not all_tasks:
        return verified

    # Capped well inside the final's 270 s limit.
    started = time.monotonic()
    stop_at = started + min(VERIFY_WALL_BUDGET_S * _budget_scale(n_commits), 200.0)

    # Run pure-Python decap (e.g. test_generator_memory) before any C extensions load.
    orig_chosen = {t.commit: set(t.chosen) for t in all_tasks}
    pre_passed = _run_pure_py_decap(all_tasks, stop_at)
    for task in all_tasks:
        pp = pre_passed.get(task.commit)
        if pp:
            task.chosen = task.chosen - pp

    phase1_todo = [t for t in phase1 if t.chosen]
    phase2_todo = [t for t in phase2 if t.chosen]

    if phase1_todo and time.monotonic() < stop_at:
        # Freeze parent heap so forked children don't trigger a full Gen-2 GC / COW storm (~0.6s/child).
        gc.freeze()
        _run_batch_parallel(phase1_todo, id_by_number, ms_by_number, cpu_by_number, out_fd, stop_at, verified, started)

    if phase2_todo and time.monotonic() < stop_at:
        by_bucket: dict[str, list[VerifyTask]] = {}
        for t in phase2_todo:
            by_bucket.setdefault(t.c_bucket, []).append(t)
        manifest = load_c_manifest() if ENABLE_MYPYC else None
        for bname in ("pure_full_c", "opt_py_c", "parse_py", "gen_py", "fallback_c"):
            btasks = by_bucket.get(bname)
            if not btasks or time.monotonic() >= stop_at:
                continue
            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r_fd)
                try:
                    os.setpgid(0, 0)
                except OSError:
                    pass
                if out_fd is not None:
                    try:
                        os.close(out_fd)
                    except OSError:
                        pass
                b_verified: dict[str, set[int]] = {}
                try:
                    os.chdir(SOURCE)
                    if str(SOURCE) not in sys.path:
                        sys.path.insert(0, str(SOURCE))
                    union_disabled = frozenset().union(*(t.disabled_c_groups for t in btasks))
                    if manifest is not None and ENABLE_MYPYC and (set(manifest["groups"]) - union_disabled):
                        install_modular_c(union_disabled, after_overlay=False)
                    install_db_cache()
                    needs_real_tpc = any(
                        _real_tpc_possible(
                            t.tpc_mode,
                            t.selection,
                            n,
                            id_by_number,
                            ms_by_number,
                            {**cpu_by_number, **t.cpu_override} if t.cpu_override else cpu_by_number,
                        )
                        for t in btasks
                        for n in t.chosen
                    )
                    if needs_real_tpc:
                        for pkg in ("numpy", "pandas"):
                            try:
                                importlib.import_module(pkg)
                            except Exception:
                                pass
                    _prewarm_for_phase2(btasks, id_by_number, ms_by_number, cpu_by_number)
                    gc.freeze()
                    _run_batch_parallel(
                        btasks,
                        id_by_number,
                        ms_by_number,
                        cpu_by_number,
                        None,
                        stop_at,
                        b_verified,
                        started,
                    )
                except BaseException as err:
                    log(f"phase2 bucket {bname} error: {err!r}")
                finally:
                    try:
                        os.write(w_fd, marshal.dumps({c: sorted(s) for c, s in b_verified.items()}))
                    except OSError:
                        pass
                    os.close(w_fd)
                    os._exit(0)
            os.close(w_fd)
            chunks = []
            while True:
                rem = (stop_at + 5.0) - time.monotonic()
                if rem <= 0 or not select.select([r_fd], [], [], min(rem, 1.0))[0]:
                    if rem <= 0:
                        break
                    continue
                chunk = os.read(r_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            _kill_group(pid)
            os.close(r_fd)
            os.waitpid(pid, 0)
            try:
                got = marshal.loads(b"".join(chunks))
                if isinstance(got, dict):
                    for c, nums in got.items():
                        verified[c] = set(nums)
            except Exception:
                pass

    for c, pp in pre_passed.items():
        verified[c] = verified.get(c, orig_chosen[c]) - pp
    return verified
