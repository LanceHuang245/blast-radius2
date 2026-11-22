"""Build-time per-case, per-subTest, per-Validator-call, and TPC-DS checkpoint precompiler.

Records fine-grained sqlglot line coverage for:
  1. TestOptimizer.check_file fixture tests (kind="check_file")
  2. TestOptimizer subTest-loop tests (kind="subtest", bucketed <=64 buckets, build-time verified)
  3. Dialect Validator tests >=25ms CPU (kind="validator", bucketed <=32 buckets, build-time verified)
  4. TPC-DS / TPC-H Stage-1 per-query line coverage (kind="tpc_stage1")

Also builds map/tpcds_ckpts.pkl with intermediate AST checkpoints across optimizer.RULES
and exact sys.setprofile-recorded cumulative file sets so TestOptimizer.test_tpcds and
verify_tpc_test("tpcds", "fast_a") skip untouched prefix stages cleanly.

Build: SKIP_INTEGRATION=0 python -m blast_radius.precompile_cases
"""

from __future__ import annotations

import ast
import glob
import importlib
import inspect
import marshal
import os
import pickle
import sys
import textwrap
import time
import unittest
from collections import defaultdict
from contextlib import contextmanager

from blast_radius import APP, SOURCE
from blast_radius.astlayout import layout as ast_layout
from blast_radius.precompile_tpc import TPC_STAGE1_FILE, TPC_STAGE2_FILE, plan_signature

CASES_FILE = APP / "map" / "cases.marshal"
TPCDS_CKPTS_META_FILE = APP / "map" / "tpcds_ckpts_meta.pkl"
TPC_S1_BLOBS_FILE = APP / "map" / "tpc_s1_blobs.pkl"
TPC_S2_BLOBS_DIR = APP / "map" / "tpc_s2_blobs"
TPCDS_CKPT_DIR = APP / "map"
OPT_TEST_FILE = "tests/test_optimizer.py"
TPCDS_CKPT_STAGES = (0, 1, 3, 5, 7, 8, 12, 14)
TPCDS_EXEC_CKPT_STAGES = (8, 9, 10, 11, 12, 13, 14)


def tpcds_ckpt_stage_file(stage: int, for_exec: bool = False):
    prefix = "tpcds_exec_ckpt_" if for_exec else "tpcds_ckpt_"
    return TPCDS_CKPT_DIR / f"{prefix}{stage}.pkl"


def tpcds_ckpt_bin_files(stage: int, for_exec: bool = False):
    prefix = "tpcds_exec_ckpt_" if for_exec else "tpcds_ckpt_"
    return (
        TPCDS_CKPT_DIR / f"{prefix}{stage}.meta",
        TPCDS_CKPT_DIR / f"{prefix}{stage}.bin",
    )


def tpc_s2_table_file(suite: str, table_name: str):
    return TPC_S2_BLOBS_DIR / suite / f"t_{table_name}.pkl"


def tpc_s2_query_file(suite: str, index: int):
    return TPC_S2_BLOBS_DIR / suite / f"q_{index}.pkl"


class _SubTestFilterTransformer(ast.NodeTransformer):
    """Wrap top-level `with self.subTest(...):` bodies in `if _br_subtest_ok():`."""

    def __init__(self) -> None:
        self._in_subtest = False
        self.wrapped = 0

    def visit_With(self, node: ast.With) -> ast.AST:
        is_sub = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "subTest"
            for item in node.items
        )
        if is_sub and not self._in_subtest:
            self._in_subtest = True
            self.generic_visit(node)
            self._in_subtest = False
            self.wrapped += 1
            guard = ast.If(
                test=ast.Call(func=ast.Name(id="_br_subtest_ok", ctx=ast.Load()), args=[], keywords=[]),
                body=node.body,
                orelse=[],
            )
            ast.copy_location(guard, node)
            node.body = [guard]
            return node
        self.generic_visit(node)
        return node


def transform_subtest_method(mod, cls: type, method_name: str, ok_fn) -> bool:
    """Replace `cls.<method_name>` in-place with a subTest-filterable compiled version."""
    fn = cls.__dict__.get(method_name)
    if fn is None or getattr(fn, "_br_subtest_wrapped", False):
        return bool(getattr(fn, "_br_subtest_wrapped", False))
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return False
    if any(isinstance(n, ast.Name) and n.id == "super" for n in ast.walk(tree)):
        return False
    transformer = _SubTestFilterTransformer()
    tree = ast.fix_missing_locations(transformer.visit(tree))
    if transformer.wrapped == 0:
        return False
    mod._br_subtest_ok = ok_fn
    local_ns: dict = {}
    exec(compile(tree, getattr(mod, "__file__", "<subtest>"), "exec"), mod.__dict__, local_ns)
    new_fn = local_ns.get(method_name)
    if new_fn is None:
        return False
    new_fn._br_subtest_wrapped = True
    setattr(cls, method_name, new_fn)
    return True


def split_independent_blocks(body: list[ast.stmt]) -> list[tuple[int, int]]:
    """Partition top-level statements into contiguous independent local-variable groups."""
    n = len(body)
    if n < 2:
        return [(0, n - 1)] if n else []

    def vars_in(node: ast.AST) -> tuple[set[str], set[str], bool]:
        reads: set[str] = set()
        writes: set[str] = set()
        mutates_state = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                if isinstance(sub.ctx, (ast.Store, ast.Del)):
                    writes.add(sub.id)
                elif isinstance(sub.ctx, ast.Load):
                    reads.add(sub.id)
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                mutates_state = True
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                mutates_state = True
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                writes.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    writes.add(alias.asname or alias.name.split(".", 1)[0])
        return reads, writes, mutates_state

    stmt_rw = [vars_in(s) for s in body]
    if any(m for _, _, m in stmt_rw):
        return [(0, n - 1)]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    last_write: dict[str, int] = {}
    for i, (reads, writes, _) in enumerate(stmt_rw):
        for var in reads:
            if var in last_write:
                for k in range(last_write[var], i + 1):
                    union(last_write[var], k)
        for var in writes:
            last_write[var] = i

    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        root = find(i)
        j = i
        while j + 1 < n and find(j + 1) == root:
            j += 1
        groups.append((i, j))
        i = j + 1
    return groups


def transform_block_method(mod, cls: type, method_name: str, ok_fn) -> int:
    """Replace `cls.<method_name>` in-place with an independent-block-filterable version; return block count."""
    fn = cls.__dict__.get(method_name)
    if fn is None:
        return 0
    prev = getattr(fn, "_br_block_count", 0)
    if prev:
        mod._br_block_ok = ok_fn
        return prev
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return 0
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return 0
    if any(isinstance(n, ast.Name) and n.id == "super" for n in ast.walk(tree)):
        return 0
    fn_node = tree.body[0]
    groups = split_independent_blocks(fn_node.body)
    if len(groups) < 2:
        return 0
    new_body: list[ast.stmt] = []
    for idx, (lo, hi) in enumerate(groups):
        guard = ast.If(
            test=ast.Call(
                func=ast.Name(id="_br_block_ok", ctx=ast.Load()),
                args=[ast.Constant(value=idx)],
                keywords=[],
            ),
            body=fn_node.body[lo : hi + 1],
            orelse=[],
        )
        ast.copy_location(guard, fn_node.body[lo])
        new_body.append(guard)
    fn_node.body = new_body
    tree = ast.fix_missing_locations(tree)
    mod._br_block_ok = ok_fn
    local_ns: dict = {}
    exec(compile(tree, getattr(mod, "__file__", "<blocks>"), "exec"), mod.__dict__, local_ns)
    new_fn = local_ns.get(method_name)
    if new_fn is None:
        return 0
    new_fn._br_block_count = len(groups)
    setattr(cls, method_name, new_fn)
    return len(groups)


def _record_block_cases(mon, tool: int, src_prefix: str, skip_tids: set[str] | None = None) -> dict[str, dict]:
    """Record per-block line coverage and CPU weights for heavy multi-block unit tests."""
    import sqlglot.generator as generator

    explicit_targets = [
        ("tests/test_optimizer.py", "TestOptimizer", "test_scope"),
        ("tests/test_optimizer.py", "TestOptimizer", "test_deep_ast_type_annotation"),
        ("tests/test_optimizer.py", "TestOptimizer", "test_values_annotation"),
        ("tests/test_expressions.py", "TestExprs", "test_is_star"),
        ("tests/test_executor.py", "TestExecutor", "test_scalar_functions"),
    ]
    skip_special = frozenset({
        "tests/test_parser.py::TestParser::test_max_nodes",
        "tests/test_generator.py::TestGenerator::test_generator_memory",
        "tests/test_executor.py::TestExecutor::test_execute_tpcds",
        "tests/test_executor.py::TestExecutor::test_execute_tpch",
        "tests/test_executor.py::TestExecutor::test_optimized_tpch",
        "tests/test_executor.py::TestExecutor::test_agg_order",
        "tests/test_serde.py::TestSerde::test_recursion",
    }) | (skip_tids or set())

    # Auto-discover additional heavy (>=50ms) multi-block tests across top-level test_*.py files.
    targets: list[tuple[str, str, str, bool]] = [(f, c, m, True) for f, c, m in explicit_targets]
    seen_targets = {(f, c, m) for f, c, m in explicit_targets}
    for rel_file in sorted(glob.glob("tests/test_*.py", root_dir=str(SOURCE))):
        if rel_file in ("tests/test_docs.py", "tests/test_dialect_imports.py", "tests/test_executor.py"):
            continue
        try:
            tree = ast.parse((SOURCE / rel_file).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name.startswith("test_")
                    and len(item.body) >= 4
                    and (rel_file, node.name, item.name) not in seen_targets
                ):
                    tid = f"{rel_file}::{node.name}::{item.name}"
                    if tid in skip_special:
                        continue
                    if len(split_independent_blocks(item.body)) >= 2:
                        targets.append((rel_file, node.name, item.name, False))
                        seen_targets.add((rel_file, node.name, item.name))

    current_lines: set[tuple[str, int]] = set()
    active = [False]
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if active[0] and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    block_state = {"mask": None}

    def _block_ok(idx: int) -> bool:
        m = block_state["mask"]
        return m is None or bool(m >> idx & 1)

    per_test: dict[str, dict] = {}
    setup_done: set[type] = set()
    try:
        for rel_file, cls_name, meth, is_explicit in targets:
            mod = importlib.import_module(rel_file[:-3].replace("/", "."))
            cls = getattr(mod, cls_name)
            if cls not in setup_done and cls_name != "TestExecutor":
                cls.setUpClass()
                setup_done.add(cls)
            if not is_explicit:
                t_chk = time.process_time()
                r_chk = unittest.TestResult()
                cls(meth).run(r_chk)
                if not r_chk.wasSuccessful() or (time.process_time() - t_chk) < 0.050:
                    continue
            n_blocks = transform_block_method(mod, cls, meth, _block_ok)
            if n_blocks < 2 or n_blocks > 64:
                continue
            cur_lines: dict[str, dict[int, int]] = defaultdict(dict)
            block_ms: list[int] = []
            ok_all = True
            for b in range(n_blocks):
                block_state["mask"] = 1 << b
                # Measure uninstrumented CPU weight for accurate ROI scaling.
                t_b = time.process_time()
                r0 = unittest.TestResult()
                cls(meth).run(r0)
                dt_ms = max(1, round((time.process_time() - t_b) * 1000))
                if not r0.wasSuccessful():
                    ok_all = False
                    break
                block_ms.append(dt_ms)
                # Record sqlglot line coverage for block b.
                generator._DISPATCH_CACHE.clear()
                current_lines.clear()
                active[0] = True
                mon.restart_events()
                mon.set_events(tool, mon.events.LINE)
                r1 = unittest.TestResult()
                cls(meth).run(r1)
                mon.set_events(tool, 0)
                active[0] = False
                if not r1.wasSuccessful():
                    ok_all = False
                    break
                bit = 1 << b
                for rel, ln in current_lines:
                    cur_lines[rel][ln] = cur_lines[rel].get(ln, 0) | bit
            if ok_all:
                tid = f"{rel_file}::{cls_name}::{meth}"
                per_test[tid] = {
                    "kind": "blocks",
                    "count": n_blocks,
                    "block_ms": block_ms,
                    "lines": dict(cur_lines),
                }
    finally:
        block_state["mask"] = None
        mon.set_events(tool, 0)
    return per_test


def _collect_used_validator_returns() -> set[tuple[str, int]]:
    """Find (abs_path, lineno) of validate_identity/validate_transpile calls whose return value is used."""
    used: set[tuple[str, int]] = set()
    for rel in sorted(glob.glob("tests/dialects/test_*.py", root_dir=str(SOURCE))):
        full = os.path.abspath(str(SOURCE / rel))
        try:
            tree = ast.parse(open(full, encoding="utf-8").read())
        except SyntaxError:
            continue
        expr_calls = {
            id(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("validate_identity", "validate_transpile")
                and id(node) not in expr_calls
            ):
                used.add((full, node.lineno))
    return used


def _record_optimizer_cases(mon, tool: int, src_prefix: str) -> dict[str, dict]:
    """Record check_file and bucketed subTest-loop per-case line coverage for TestOptimizer."""
    from blast_radius import verify as V
    import sqlglot.generator as generator
    import tests.test_optimizer as T

    # Must patch tests.test_optimizer.wait / as_completed AFTER importing it, otherwise
    # concurrent.futures.wait deadlocks forever on _LazyFuture.
    V._patch_pool_references()

    state = {
        "in_check": None,
        "pending": [],
        "depth": 0,
        "case": None,
        "mode": None,  # "check_file" | "subtest" | "count"
        "broken": False,
        "recorder": None,
        "sub_idx": 0,
        "stride": 1,
        "cur_bucket": -1,
    }
    current_lines: set[tuple[str, int]] = set()
    per_test: dict[str, dict] = {}
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if state["case"] is not None and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)

    orig_loader = T.load_sql_fixture_pairs

    def recording_loader(filename):
        for index, item in enumerate(orig_loader(filename)):
            if state["in_check"] is not None:
                state["pending"].append((filename, index))
            yield item

    T.load_sql_fixture_pairs = recording_loader

    orig_check_file = T.TestOptimizer.check_file

    def recording_check_file(self, file, *args, **kwargs):
        state["in_check"], state["pending"] = file, []
        try:
            return orig_check_file(self, file, *args, **kwargs)
        finally:
            if state["pending"]:
                state["broken"] = True
            state["in_check"] = None

    T.TestOptimizer.check_file = recording_check_file

    orig_subtest = unittest.TestCase.subTest

    @contextmanager
    def recording_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        mode = state["mode"]
        top_cf = mode == "check_file" and state["in_check"] is not None and state["depth"] == 0
        top_st = mode == "subtest" and state["in_check"] is None and state["depth"] == 0
        if mode == "count" and state["depth"] == 0:
            state["sub_idx"] += 1
        elif top_cf:
            if not state["pending"]:
                state["broken"] = True
                with orig_subtest(self, msg, **params):
                    yield
                return
            case = state["pending"].pop(0)
            generator._DISPATCH_CACHE.clear()
            current_lines.clear()
            state["case"] = case
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
        elif top_st:
            idx = state["sub_idx"]
            state["sub_idx"] = idx + 1
            bucket = idx // state["stride"]
            if bucket != state["cur_bucket"]:
                if state["cur_bucket"] >= 0:
                    mon.set_events(tool, 0)
                    state["recorder"](state["cur_bucket"], set(current_lines))
                generator._DISPATCH_CACHE.clear()
                current_lines.clear()
                state["cur_bucket"] = bucket
                state["case"] = bucket
                mon.restart_events()
                mon.set_events(tool, mon.events.LINE)
        state["depth"] += 1
        try:
            with orig_subtest(self, msg, **params):
                yield
        finally:
            state["depth"] -= 1
            if top_cf:
                mon.set_events(tool, 0)
                state["recorder"](state["case"], set(current_lines))
                state["case"] = None

    unittest.TestCase.subTest = recording_subtest

    cls = T.TestOptimizer
    cls.setUpClass()
    cls_tree = ast.parse(inspect.getsource(cls))
    method_kinds: dict[str, str] = {}
    for node in cls_tree.body[0].body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            has_cf = any(isinstance(n, ast.Attribute) and n.attr == "check_file" for n in ast.walk(node))
            has_sub = any(isinstance(n, ast.Attribute) and n.attr == "subTest" for n in ast.walk(node))
            if has_cf:
                method_kinds[node.name] = "check_file"
            elif has_sub:
                method_kinds[node.name] = "subtest"

    verify_state = {"mask": None, "idx": 0, "stride": 1}

    def _verify_subtest_ok() -> bool:
        idx = verify_state["idx"]
        verify_state["idx"] = idx + 1
        mask = verify_state["mask"]
        return mask is None or bool(mask >> (idx // verify_state["stride"]) & 1)

    for name, kind in sorted(method_kinds.items()):
        tid = f"{OPT_TEST_FILE}::TestOptimizer::{name}"
        if kind == "check_file":
            cases: list = []
            lines: dict[str, dict[int, int]] = defaultdict(dict)

            def record_cf(case, executed, cases=cases, lines=lines):
                bit = 1 << len(cases)
                cases.append(case)
                for rel, line in executed:
                    lines[rel][line] = lines[rel].get(line, 0) | bit

            state.update(mode="check_file", recorder=record_cf, broken=False, depth=0, case=None)
            result = unittest.TestResult()
            cls(name).run(result)
            if cases and not state["broken"] and result.wasSuccessful():
                per_test[tid] = {"kind": "check_file", "cases": cases, "lines": dict(lines)}
            continue

        # SubTest loop: first count subtests & time uninstrumented; skip tiny (<20ms) tests.
        state.update(mode="count", depth=0, sub_idx=0, case=None, cur_bucket=-1)
        t_chk = time.process_time()
        r_chk = unittest.TestResult()
        cls(name).run(r_chk)
        total_sub = state["sub_idx"]
        if not r_chk.wasSuccessful() or total_sub < 2 or (time.process_time() - t_chk) < 0.020:
            continue

        stride = max(1, (total_sub + 63) // 64)
        n_buckets = (total_sub + stride - 1) // stride
        lines = defaultdict(dict)

        def record_st(bucket, executed, lines=lines):
            bit = 1 << bucket
            for rel, line in executed:
                lines[rel][line] = lines[rel].get(line, 0) | bit

        state.update(
            mode="subtest",
            recorder=record_st,
            broken=False,
            depth=0,
            case=None,
            sub_idx=0,
            stride=stride,
            cur_bucket=-1,
        )
        result = unittest.TestResult()
        cls(name).run(result)
        mon.set_events(tool, 0)
        if state["cur_bucket"] >= 0:
            record_st(state["cur_bucket"], set(current_lines))
        state["case"] = None
        if not result.wasSuccessful() or state["sub_idx"] != total_sub:
            continue

        # Self-verify transformed method with first bucket and last bucket only.
        unittest.TestCase.subTest = orig_subtest
        try:
            if not transform_subtest_method(T, cls, name, _verify_subtest_ok):
                continue
            ok_both = True
            for b_probe in (0, n_buckets - 1):
                verify_state.update(mask=1 << b_probe, idx=0, stride=stride)
                r = unittest.TestResult()
                cls(name).run(r)
                if not r.wasSuccessful() or verify_state["idx"] != total_sub:
                    ok_both = False
                    break
            if ok_both:
                per_test[tid] = {
                    "kind": "subtest",
                    "count": n_buckets,
                    "stride": stride,
                    "lines": dict(lines),
                }
        finally:
            verify_state["mask"] = None
            unittest.TestCase.subTest = recording_subtest

    mon.set_events(tool, 0)
    T.load_sql_fixture_pairs = orig_loader
    T.TestOptimizer.check_file = orig_check_file
    unittest.TestCase.subTest = orig_subtest
    return per_test


def _record_validator_cases(mon, tool: int, src_prefix: str) -> dict[str, dict]:
    """Record bucketed per-call coverage for Validator dialect tests >=25ms CPU."""
    import sqlglot.generator as generator
    from tests.dialects.test_dialect import Validator

    used_ret = _collect_used_validator_returns()
    state = {
        "active": False,
        "idx": 0,
        "depth": 0,
        "always": 0,
        "skip_mask": None,
        "stride": 1,
        "cur_bucket": -1,
    }
    current_lines: set[tuple[str, int]] = set()
    lines_ref: list[dict[str, dict[int, int]]] = [{}]
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if state["active"] and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)

    orig_id = Validator.validate_identity
    orig_all = Validator.validate_all
    orig_tr = Validator.validate_transpile

    def flush_bucket() -> None:
        b = state["cur_bucket"]
        if b >= 0:
            mon.set_events(tool, 0)
            bit = 1 << b
            cur_map = lines_ref[0]
            for rel, ln in current_lines:
                cur_map[rel][ln] = cur_map[rel].get(ln, 0) | bit
            state["cur_bucket"] = -1
            state["active"] = False

    def make_wrapper(orig, can_return: bool):
        def wrapped(self, *args, **kwargs):
            if state["depth"] > 0:
                return orig(self, *args, **kwargs)
            idx = state["idx"]
            state["idx"] = idx + 1
            bucket = idx // state["stride"]
            if state["skip_mask"] is not None:
                if not (state["skip_mask"] >> bucket & 1):
                    return None
                state["depth"] += 1
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    state["depth"] -= 1
            if can_return:
                frame = sys._getframe(1)
                if (os.path.abspath(frame.f_code.co_filename), frame.f_lineno) in used_ret:
                    state["always"] |= 1 << bucket
            if bucket != state["cur_bucket"]:
                flush_bucket()
                generator._DISPATCH_CACHE.clear()
                current_lines.clear()
                state["cur_bucket"] = bucket
                state["active"] = True
                mon.restart_events()
                mon.set_events(tool, mon.events.LINE)
            state["depth"] += 1
            try:
                return orig(self, *args, **kwargs)
            finally:
                state["depth"] -= 1

        return wrapped

    Validator.validate_identity = make_wrapper(orig_id, True)
    Validator.validate_all = make_wrapper(orig_all, False)
    Validator.validate_transpile = make_wrapper(orig_tr, True)

    per_test: dict[str, dict] = {}
    try:
        for rel_file in sorted(glob.glob("tests/dialects/test_*.py", root_dir=str(SOURCE))):
            mod_name = rel_file[:-3].replace("/", ".")
            mod = importlib.import_module(mod_name)
            for cls_name, cls in sorted(vars(mod).items()):
                if not (isinstance(cls, type) and issubclass(cls, Validator) and cls is not Validator):
                    continue
                if cls.__module__ != mod_name:
                    continue
                for meth in sorted(dir(cls)):
                    if not meth.startswith("test_") or meth == "test_lazy_load":
                        continue
                    # Uninstrumented timing pass: only map dialect tests >=25ms CPU with >=5 calls.
                    state.update(active=False, idx=0, depth=0, always=0, skip_mask=-1, stride=1, cur_bucket=-1)
                    t_chk = time.process_time()
                    r_chk = unittest.TestResult()
                    cls(meth).run(r_chk)
                    total_calls = state["idx"]
                    if not r_chk.wasSuccessful() or (time.process_time() - t_chk) < 0.025 or total_calls < 5:
                        continue

                    stride = max(1, (total_calls + 31) // 32)
                    n_buckets = (total_calls + stride - 1) // stride
                    cur_lines: dict[str, dict[int, int]] = defaultdict(dict)
                    lines_ref[0] = cur_lines
                    state.update(active=False, idx=0, depth=0, always=0, skip_mask=None, stride=stride, cur_bucket=-1)
                    res = unittest.TestResult()
                    cls(meth).run(res)
                    flush_bucket()
                    always = state["always"]
                    if not res.wasSuccessful() or state["idx"] != total_calls:
                        continue

                    # Self-verify: running only `always` buckets must pass with identical call count.
                    state.update(active=False, idx=0, depth=0, skip_mask=always, stride=stride, cur_bucket=-1)
                    res2 = unittest.TestResult()
                    cls(meth).run(res2)
                    if res2.wasSuccessful() and state["idx"] == total_calls:
                        tid = f"{rel_file}::{cls_name}::{meth}"
                        per_test[tid] = {
                            "kind": "validator",
                            "count": n_buckets,
                            "stride": stride,
                            "always": always,
                            "lines": dict(cur_lines),
                        }
    finally:
        mon.set_events(tool, 0)
        Validator.validate_identity = orig_id
        Validator.validate_all = orig_all
        Validator.validate_transpile = orig_tr

    return per_test


def _record_extra_subtest_cases(mon, tool: int, src_prefix: str) -> dict[str, dict]:
    """Record bucketed subTest coverage for heavy non-optimizer tests (test_identity, test_cache_does_not_retain_queries)."""
    import sqlglot.generator as generator

    targets = (
        ("tests/test_transpile.py", "TestTranspile", "test_identity"),
        ("tests/test_schema.py", "TestSchema", "test_cache_does_not_retain_queries"),
        ("tests/test_serde.py", "TestSerde", "test_serde"),
        ("tests/test_optimize_cross_joins.py", "TestOptimizeCrossJoins", "test_conditional_cross_joins"),
        ("tests/test_expressions.py", "TestExprs", "test_named_selects_long_by_name_chain"),
        ("tests/dialects/test_dialect.py", "TestDialect", "test_initcap"),
        ("tests/test_parser.py", "TestParser", "test_is_precedence"),
        ("tests/test_build.py", "TestBuild", "test_build"),
        ("tests/test_simplify_between.py", "TestSimplifyBetween", "test_symmetric_truth_values_and_grouping"),
        ("tests/test_jsonpath.py", "TestJsonpath", "test_cts_file"),
        ("tests/dialects/test_clickhouse.py", "TestClickhouse", "test_json_extract_qualified_roundtrip"),
        ("tests/dialects/test_dialect.py", "TestDialect", "test_decimal_literals"),
    )
    prefix_cut = len(src_prefix) - len("sqlglot/")
    state = {"active": False, "mode": None, "depth": 0, "sub_idx": 0, "stride": 1, "cur_bucket": -1}
    current_lines: set[tuple[str, int]] = set()
    lines_ref: list[dict[str, dict[int, int]]] = [{}]

    def on_line(code, line):
        if state["active"] and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    orig_subtest = unittest.TestCase.subTest

    def flush() -> None:
        b = state["cur_bucket"]
        if b >= 0:
            mon.set_events(tool, 0)
            bit = 1 << b
            cur_map = lines_ref[0]
            for rel, ln in current_lines:
                cur_map[rel][ln] = cur_map[rel].get(ln, 0) | bit
            state["cur_bucket"] = -1
            state["active"] = False

    @contextmanager
    def recording_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        if state["depth"] == 0:
            if state["mode"] == "count":
                state["sub_idx"] += 1
            elif state["mode"] == "subtest":
                idx = state["sub_idx"]
                state["sub_idx"] = idx + 1
                bucket = idx // state["stride"]
                if bucket != state["cur_bucket"]:
                    flush()
                    generator._DISPATCH_CACHE.clear()
                    current_lines.clear()
                    state["cur_bucket"] = bucket
                    state["active"] = True
                    mon.restart_events()
                    mon.set_events(tool, mon.events.LINE)
        state["depth"] += 1
        try:
            with orig_subtest(self, msg, **params):
                yield
        finally:
            state["depth"] -= 1

    verify_state = {"mask": None, "idx": 0, "stride": 1}

    def _verify_ok() -> bool:
        idx = verify_state["idx"]
        verify_state["idx"] = idx + 1
        mask = verify_state["mask"]
        return mask is None or bool(mask >> (idx // verify_state["stride"]) & 1)

    per_test: dict[str, dict] = {}
    unittest.TestCase.subTest = recording_subtest
    try:
        for rel_file, cls_name, meth in targets:
            mod = importlib.import_module(rel_file[:-3].replace("/", "."))
            cls = getattr(mod, cls_name)
            state.update(active=False, mode="count", depth=0, sub_idx=0, stride=1, cur_bucket=-1)
            r0 = unittest.TestResult()
            cls(meth).run(r0)
            total_sub = state["sub_idx"]
            if not r0.wasSuccessful() or total_sub < 2:
                continue
            stride = max(1, (total_sub + 31) // 32)
            n_buckets = (total_sub + stride - 1) // stride
            cur_lines: dict[str, dict[int, int]] = defaultdict(dict)
            lines_ref[0] = cur_lines
            state.update(active=False, mode="subtest", depth=0, sub_idx=0, stride=stride, cur_bucket=-1)
            r1 = unittest.TestResult()
            cls(meth).run(r1)
            flush()
            if not r1.wasSuccessful() or state["sub_idx"] != total_sub:
                continue
            unittest.TestCase.subTest = orig_subtest
            try:
                if not transform_subtest_method(mod, cls, meth, _verify_ok):
                    continue
                ok_both = True
                for b_probe in (0, n_buckets - 1):
                    verify_state.update(mask=1 << b_probe, idx=0, stride=stride)
                    r2 = unittest.TestResult()
                    cls(meth).run(r2)
                    if not r2.wasSuccessful() or verify_state["idx"] != total_sub:
                        ok_both = False
                        break
                if ok_both:
                    tid = f"{rel_file}::{cls_name}::{meth}"
                    per_test[tid] = {
                        "kind": "subtest",
                        "count": n_buckets,
                        "stride": stride,
                        "lines": dict(cur_lines),
                    }
            finally:
                verify_state["mask"] = None
                unittest.TestCase.subTest = recording_subtest
    finally:
        mon.set_events(tool, 0)
        unittest.TestCase.subTest = orig_subtest
    return per_test


def _record_tpc_and_ckpts(mon, tool: int, src_prefix: str) -> tuple[dict[str, dict], dict]:
    """Record per-query TPC Stage-1 & Stage-2b coverage, fast per-query/table blobs, and TPC-DS checkpoints."""
    from sqlglot import Dialect, find_tables, parse_one
    from sqlglot.executor.python import Python, PythonExecutor
    from sqlglot.executor.table import Table, ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.optimizer import RULES
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump, load as serde_load
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs
    import sqlglot.generator as generator

    per_tpc: dict[str, dict] = {}
    s1_all = pickle.load(open(TPC_STAGE1_FILE, "rb")) if TPC_STAGE1_FILE.is_file() else {}
    s2_all = pickle.load(open(TPC_STAGE2_FILE, "rb")) if TPC_STAGE2_FILE.is_file() else {}
    cur_lines: set[tuple[str, int]] = set()
    recording = [False]
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if recording[0] and code.co_filename.startswith(src_prefix):
            cur_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    duck_gen = Dialect.get_or_raise("duckdb").generator()
    py_gen = Python().generator(identify=True, comments=False)

    # Convert monolithic tpc_stage1.pkl & tpc_stage2.pkl into lazy per-query / per-table blobs
    # so children load in ~2-5ms instead of unpickling 33MB (~1.2s).
    s1_blobs: dict[str, dict] = {}
    s2b_query_ms_by_suite: dict[str, list[int]] = {}
    for suite, tid in (
        ("tpcds", "tests/test_executor.py::TestExecutor::test_execute_tpcds"),
        ("tpch", "tests/test_executor.py::TestExecutor::test_execute_tpch"),
    ):
        s1 = s1_all.get(suite)
        if not s1:
            continue
        ensured = ensure_schema(s1["schema"])
        q_lines: dict[str, dict[int, int]] = defaultdict(dict)
        for idx, q in enumerate(s1["queries"]):
            generator._DISPATCH_CACHE.clear()
            cur_lines.clear()
            recording[0] = True
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
            expr = parse_one(q["sql"])
            if q["execute"]:
                _ = {t.name for t in find_tables(expr)}
                _ = duck_gen.generate(expr)
                opt_expr = optimize(expr, ensured, leave_tables_isolated=True)
                _ = plan_signature(Plan(opt_expr), serde_dump)
            mon.set_events(tool, 0)
            recording[0] = False
            bit = 1 << idx
            for rel, ln in cur_lines:
                q_lines[rel][ln] = q_lines[rel].get(ln, 0) | bit

        generator._DISPATCH_CACHE.clear()
        cur_lines.clear()
        recording[0] = True
        mon.restart_events()
        mon.set_events(tool, mon.events.LINE)
        for expr_dump, _ in s1["gen_pairs"]:
            py_gen.generate(serde_load(expr_dump))
        mon.set_events(tool, 0)
        recording[0] = False
        s2a_lines: dict[str, list[int]] = defaultdict(list)
        for rel, ln in sorted(cur_lines):
            s2a_lines[rel].append(ln)

        # Record per-query Stage-2b coverage (PythonExecutor.execute) and write per-table/query blobs.
        # Use local LINE events on stage2b_files code objects only so dynamically eval'd <string>
        # SQL expressions run at full native speed during Docker build.
        s2b_lines: dict[str, dict[int, int]] = defaultdict(dict)
        s2 = s2_all.get(suite)
        if s2:
            suite_dir = TPC_S2_BLOBS_DIR / suite
            suite_dir.mkdir(parents=True, exist_ok=True)
            with open(suite_dir / "meta.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "schema": s2["schema"],
                        "n_queries": len(s2["queries"]),
                        "exec_tables": [q.get("table_names", []) if q["execute"] else [] for q in s2["queries"]],
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            for tname, rows in s2["tables_raw"].items():
                with open(tpc_s2_table_file(suite, tname), "wb") as handle:
                    pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tables_by_name = {
                name: Table(columns=s2["schema"][name], rows=list(rows))
                for name, rows in s2["tables_raw"].items()
            }
            # Collect code objects from sqlglot/executor/* (the Stage-2b runtime modules).
            s2b_codes: list = []
            seen_codes: set[int] = set()

            def _collect_codes(co) -> None:
                if id(co) in seen_codes:
                    return
                seen_codes.add(id(co))
                if getattr(co, "co_filename", "").startswith(src_prefix + "executor/"):
                    s2b_codes.append(co)
                for const in getattr(co, "co_consts", ()):
                    if hasattr(const, "co_code"):
                        _collect_codes(const)

            import types as _types

            for mod_name, mod_obj in list(sys.modules.items()):
                if mod_name.startswith("sqlglot.executor") and isinstance(mod_obj, _types.ModuleType):
                    for obj in vars(mod_obj).values():
                        if hasattr(obj, "__code__"):
                            _collect_codes(obj.__code__)
                        elif isinstance(obj, type):
                            for attr in vars(obj).values():
                                fn = getattr(attr, "__func__", attr)
                                if hasattr(fn, "__code__"):
                                    _collect_codes(fn.__code__)

            s2b_query_ms: list[int] = [0] * len(s2["queries"])
            for idx, q in enumerate(s2["queries"]):
                with open(tpc_s2_query_file(suite, idx), "wb") as handle:
                    pickle.dump(q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if not q["execute"]:
                    continue
                t_exec = time.process_time()
                opt_expr = serde_load(q["opt_dump"])
                plan = Plan(opt_expr)
                q_tables = {name: tables_by_name[name] for name in q["table_names"]}
                cur_lines.clear()
                recording[0] = True
                mon.restart_events()
                for co in s2b_codes:
                    mon.set_local_events(tool, co, mon.events.LINE)
                tables_ = ensure_tables(q_tables)
                PythonExecutor(tables=tables_).execute(plan)
                s2b_query_ms[idx] = max(1, round((time.process_time() - t_exec) * 1000))
                for co in s2b_codes:
                    mon.set_local_events(tool, co, 0)
                recording[0] = False
                bit = 1 << idx
                for rel, ln in cur_lines:
                    s2b_lines[rel][ln] = s2b_lines[rel].get(ln, 0) | bit
            s2b_query_ms_by_suite[suite] = s2b_query_ms
            with open(suite_dir / "meta.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "schema": s2["schema"],
                        "n_queries": len(s2["queries"]),
                        "exec_tables": [q.get("table_names", []) if q["execute"] else [] for q in s2["queries"]],
                        "s2b_query_ms": s2b_query_ms,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        # Store Stage-1 per-query opt_dump & gen_pairs as lazy bytes blobs.
        q_blobs = []
        for q in s1["queries"]:
            qb = dict(q)
            if "opt_dump" in qb:
                qb["opt_dump_b"] = pickle.dumps(qb.pop("opt_dump"), protocol=4)
            q_blobs.append(qb)
        s1_blobs[suite] = {
            "name": s1["name"],
            "schema": s1["schema"],
            "queries": q_blobs,
            "gen_pairs_b": pickle.dumps(s1["gen_pairs"], protocol=4),
        }

        per_tpc[tid] = {
            "kind": "tpc_stage1",
            "count": len(s1["queries"]),
            "lines": dict(q_lines),
            "s2a_lines": dict(s2a_lines),
            "s2b_lines": dict(s2b_lines),
        }

    with open(TPC_S1_BLOBS_FILE, "wb") as handle:
        pickle.dump(s1_blobs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mon.set_events(tool, 0)

    # Build TPC-DS rule checkpoints (Stages 0, 1, 3, 5, 7, 8, 12, 14) with both file-level sets
    # and exact per-line stage bitmasks via sys.monitoring (mon.DISABLE -> fast & accurate timing).
    pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    sqls = [sql for _, sql, _ in pairs]
    expected = [exp_sql for _, _, exp_sql in pairs]
    kw = {
        "db": None,
        "catalog": None,
        "schema": None,
        "dialect": None,
        "sql": None,
        "isolate_tables": True,
        "quote_identifiers": False,
    }
    stage_lines: dict[str, dict[int, int]] = defaultdict(dict)

    def _run_stage(bit_idx: int, fn) -> tuple[object, set[str], float]:
        generator._DISPATCH_CACHE.clear()
        cur_lines.clear()
        recording[0] = True
        mon.restart_events()
        mon.set_events(tool, mon.events.LINE)
        t_s = time.process_time()
        val = fn()
        dt_ms = (time.process_time() - t_s) * 1000.0
        mon.set_events(tool, 0)
        recording[0] = False
        bit = 1 << bit_idx
        files: set[str] = set()
        for rel, ln in cur_lines:
            stage_lines[rel][ln] = stage_lines[rel].get(ln, 0) | bit
            files.add(rel)
        return val, files, dt_ms

    base_opt_files: set[str] = {
        "sqlglot/optimizer/optimizer.py",
        "sqlglot/schema.py",
        "sqlglot/serde.py",
    }
    cum_files = set(base_opt_files)
    rule_times_ms: list[float] = []
    rule_files: list[set[str]] = []
    ckpt_data: dict[int, dict] = {}

    # Bit 21: ensure_schema (blocks skipping when schema construction lines are edited).
    schema, schema_files, _ = _run_stage(21, lambda: ensure_schema(TPCDS_SCHEMA))
    kw["schema"] = schema
    cum_files |= schema_files

    # Bit 0: parse_one(sql) for all 99 queries.
    # Freeze dumps to bytes immediately so later rules mutating expr.meta in-place cannot alter checkpoints!
    exprs, p_files, _ = _run_stage(0, lambda: [parse_one(sql) for sql in sqls])
    cum_files |= p_files
    parse_files = set(p_files) | schema_files | {"sqlglot/serde.py"}
    ckpt_data[0] = {
        "files": sorted(cum_files),
        "dumps_b": [pickle.dumps(serde_dump(e), protocol=4) for e in exprs],
        "layouts_b": [pickle.dumps(ast_layout(e), protocol=4) for e in exprs],
    }

    # Bits 1..14: RULES[0..13].
    for idx, rule in enumerate(RULES):
        if idx in TPCDS_CKPT_STAGES and idx not in ckpt_data:
            ckpt_data[idx] = {
                "files": sorted(cum_files),
                "dumps_b": [pickle.dumps(serde_dump(e), protocol=4) for e in exprs],
                "layouts_b": [pickle.dumps(ast_layout(e), protocol=4) for e in exprs],
            }
        params = inspect.getfullargspec(rule).args
        if idx < 7:
            assert "leave_tables_isolated" not in params
        rkw = {p: kw[p] for p in params if p in kw}
        exprs, r_files, r_ms = _run_stage(idx + 1, lambda r=rule, rk=rkw, es=exprs: [r(e, **rk) for e in es])
        rule_times_ms.append(r_ms)
        rule_files.append(r_files | base_opt_files | schema_files)
        cum_files |= r_files

    ckpt_data[14] = {
        "files": sorted(cum_files),
        "dumps_b": [pickle.dumps(serde_dump(e), protocol=4) for e in exprs],
        "layouts_b": [pickle.dumps(ast_layout(e), protocol=4) for e in exprs],
    }

    # Bit 17: e.sql(pretty=True) for verify_opt_tpcds.
    _, g_files, sql_ms = _run_stage(17, lambda: [e.sql(pretty=True) for e in exprs])
    gen_files = g_files | {"sqlglot/serde.py"}

    # Bits 22..28: RULES[7..13] with leave_tables_isolated=True on the 57 execute=True queries,
    # saving per-stage exec checkpoints (8..14) so fast_a starts/stops at any rule!
    exec_tail_files: set[str] = set()
    exec_ckpt_data: dict[int, dict] = {}
    exec_rule_times_ms: list[float] = []
    exec_rule_files: list[set[str]] = []
    s1_tpcds_pre = s1_all.get("tpcds")
    exec_indices = (
        [i for i, q in enumerate(s1_tpcds_pre["queries"]) if q["execute"]]
        if s1_tpcds_pre
        else []
    )
    if 7 in ckpt_data and exec_indices:
        exec_exprs = [serde_load(pickle.loads(ckpt_data[7]["dumps_b"][i])) for i in exec_indices]
        exec_kw = {**kw, "leave_tables_isolated": True}
        exec_cum_files = set(ckpt_data[7]["files"])
        n_q = len(sqls)

        def _pack_exec_ckpt(cur_list: list) -> tuple[list, list]:
            dumps_out: list = [None] * n_q
            layouts_out: list = [None] * n_q
            for q_idx, e_obj in zip(exec_indices, cur_list):
                dumps_out[q_idx] = pickle.dumps(serde_dump(e_obj), protocol=4)
                layouts_out[q_idx] = pickle.dumps(ast_layout(e_obj), protocol=4)
            return dumps_out, layouts_out

        for r_idx in range(7, len(RULES)):
            rule = RULES[r_idx]
            params = inspect.getfullargspec(rule).args
            rkw = {p: exec_kw[p] for p in params if p in exec_kw}
            exec_exprs, er_files, er_ms = _run_stage(
                22 + (r_idx - 7),
                lambda r=rule, rk=rkw, es=exec_exprs: [r(e, **rk) for e in es],
            )
            exec_rule_times_ms.append(er_ms)
            exec_rule_files.append(er_files | base_opt_files | schema_files)
            exec_tail_files |= er_files
            exec_cum_files |= er_files
            stage_k = r_idx + 1
            if stage_k in TPCDS_EXEC_CKPT_STAGES:
                d_out, l_out = _pack_exec_ckpt(exec_exprs)
                exec_ckpt_data[stage_k] = {
                    "files": sorted(exec_cum_files),
                    "dumps_b": d_out,
                    "layouts_b": l_out,
                }
        exec_tail_files |= base_opt_files | schema_files

    # Also record exact tokenizer lines exercised by test_max_nodes's SELECT x=x=x pattern.
    from sqlglot import Dialect as _Dialect

    _tok_d = _Dialect.get_or_raise(None)
    cur_lines.clear()
    recording[0] = True
    mon.restart_events()
    mon.set_events(tool, mon.events.LINE)
    _tok_d.tokenize("SELECT x=x=x,x=x=x,x=x=x")
    mon.set_events(tool, 0)
    recording[0] = False
    max_nodes_tok_lines: dict[str, list[int]] = defaultdict(list)
    for rel, ln in sorted(cur_lines):
        if rel in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py", "sqlglot/trie.py"):
            max_nodes_tok_lines[rel].append(ln)

    s1_tpcds = s1_all.get("tpcds")
    s1_duck_files: set[str] = set()
    s1_plan_files: set[str] = set()
    s1_pygen_files: set[str] = set()
    if s1_tpcds:
        # Bit 18: duck_gen.generate(parse_one(sql)).
        _, s1_duck_files, _ = _run_stage(
            18,
            lambda: [duck_gen.generate(parse_one(q["sql"])) for q in s1_tpcds["queries"] if q["execute"]],
        )
        # Bit 16: Plan(opt_expr).
        _, s1_plan_files, _ = _run_stage(
            16,
            lambda: [Plan(serde_load(q["opt_dump"])) for q in s1_tpcds["queries"] if q["execute"]],
        )
        # Bit 19: py_gen.generate(expr).
        _, s1_pygen_files, _ = _run_stage(
            19,
            lambda: [py_gen.generate(serde_load(expr_dump)) for expr_dump, _ in s1_tpcds["gen_pairs"]],
        )
        # Bit 20: find_tables(expr).
        parsed_exec = [
            serde_load(pickle.loads(ckpt_data[0]["dumps_b"][i]))
            for i, q in enumerate(s1_tpcds["queries"])
            if q["execute"]
        ]
        _run_stage(20, lambda: [{t.name for t in find_tables(e)} for e in parsed_exec])

    for k, entry in ckpt_data.items():
        rem_ms = (sum(rule_times_ms[k:]) * 0.85) + sql_ms + 40.0
        entry["est_ms"] = max(80, round(rem_ms))
        suffix_opt: set[str] = set()
        for rf in rule_files[k:]:
            suffix_opt |= rf
        entry["suffix_files"] = sorted(suffix_opt)
        entry["suffix_exec_files"] = sorted(suffix_opt | exec_tail_files) if k <= 7 else sorted(suffix_opt)

    for k, entry in exec_ckpt_data.items():
        rem_exec_ms = sum(exec_rule_times_ms[k - 7 :]) * 0.85 + 35.0
        entry["est_ms"] = max(60, round(rem_exec_ms))
        suffix_exec: set[str] = set()
        for rf in exec_rule_files[k - 7 :]:
            suffix_exec |= rf
        entry["suffix_exec_files"] = sorted(suffix_exec)

    ckpts_payload = {
        "sqls": sqls,
        "expected": expected,
        "schema": TPCDS_SCHEMA,
        "checkpoints": ckpt_data,
        "exec_checkpoints": exec_ckpt_data,
        "stage_lines": dict(stage_lines),
        "s2b_query_ms": s2b_query_ms_by_suite,
        "rule_times_ms": [round(x, 1) for x in rule_times_ms],
        "exec_rule_times_ms": [round(x, 1) for x in exec_rule_times_ms],
        "sql_ms": round(sql_ms, 1),
        "gen_files": sorted(gen_files),
        "parse_files": sorted(parse_files),
        "s1_opt_files": sorted(cum_files | exec_tail_files),
        "s1_duck_files": sorted(s1_duck_files | {"sqlglot/serde.py"}),
        "s1_plan_files": sorted(s1_plan_files | {"sqlglot/serde.py"}),
        "s1_pygen_files": sorted(s1_pygen_files | {"sqlglot/serde.py"}),
        "max_nodes_tok_lines": dict(max_nodes_tok_lines),
    }
    return per_tpc, ckpts_payload


def _record_db_cache_child() -> int:
    """Run DB-backed tests under install_db_recorder() in an isolated child and write db_cache.pkl."""
    from blast_radius import verify as V
    from blast_radius.db_cache import install_db_recorder, restore_orig_db_connect, save_db_cache

    install_db_recorder()
    V._SKIP_POOL_PICKLE = True
    V._patch_pool_references()

    import tests.test_optimizer as T_opt
    import tests.test_optimize_cross_joins as T_cj
    import tests.test_simplify_between as T_sb
    import tests.dialects.test_sqlite as T_sq

    V._patch_pool_references()
    T_opt.TestOptimizer.setUpClass()
    try:
        src_opt = (SOURCE / "tests" / "test_optimizer.py").read_text(encoding="utf-8")
        tree_opt = ast.parse(src_opt)
        lines_opt = src_opt.splitlines()
        for node in ast.walk(tree_opt):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                body = "\n".join(lines_opt[node.lineno - 1 : node.end_lineno])
                if any(k in body for k in ("duckdb.connect", "self.conn", "sqlite3.connect", "execute=True")):
                    r = unittest.TestResult()
                    T_opt.TestOptimizer(node.name).run(r)

        for cls in (T_cj.TestOptimizeCrossJoins, T_sb.TestSimplifyBetween, T_sq.TestSQLite):
            if hasattr(cls, "setUpClass"):
                cls.setUpClass()
            for name in sorted(dir(cls)):
                if name.startswith("test_"):
                    r = unittest.TestResult()
                    cls(name).run(r)
        return save_db_cache()
    finally:
        restore_orig_db_connect()


def _write_ckpt_bin(stage: int, entry: dict, expected: list, schema: dict, sqls: list | None, for_exec: bool) -> int:
    """Write offset-indexed .meta + .bin files so children only unpickle needed query indices."""
    meta_path, bin_path = tpcds_ckpt_bin_files(stage, for_exec=for_exec)
    dumps_b = entry["dumps_b"]
    layouts_b = entry["layouts_b"]
    offsets: list[tuple[int, int] | None] = []
    cur_off = 0
    with open(bin_path, "wb") as bf:
        for db, lb in zip(dumps_b, layouts_b):
            if db is None:
                offsets.append(None)
                continue
            rec = pickle.dumps((db, lb), protocol=pickle.HIGHEST_PROTOCOL)
            bf.write(rec)
            offsets.append((cur_off, len(rec)))
            cur_off += len(rec)
    meta = {
        "offsets": offsets,
        "expected": expected,
        "schema": schema,
    }
    if sqls is not None:
        meta["sqls"] = sqls
    with open(meta_path, "wb") as mf:
        pickle.dump(meta, mf, protocol=pickle.HIGHEST_PROTOCOL)
    return bin_path.stat().st_size + meta_path.stat().st_size


def _intern_cases(all_cases: dict[str, dict]) -> dict:
    """Intern bitmasks in cases.marshal into a shared little-endian bytes table."""
    seen: dict[int, int] = {}
    sets_bytes: list[bytes] = []

    def intern(m: int) -> int:
        idx = seen.get(m)
        if idx is None:
            idx = len(sets_bytes)
            seen[m] = idx
            sets_bytes.append(m.to_bytes((m.bit_length() + 7) // 8, "little"))
        return idx

    compacted: dict[str, dict] = {}
    for tid, entry in all_cases.items():
        ne = dict(entry)
        if "lines" in ne:
            ne["lines"] = {
                rel: {ln: intern(m) for ln, m in per.items()}
                for rel, per in ne["lines"].items()
            }
        if "s2b_lines" in ne:
            ne["s2b_lines"] = {
                rel: {ln: intern(m) for ln, m in per.items()}
                for rel, per in ne["s2b_lines"].items()
            }
        compacted[tid] = ne
    return {"interned": True, "sets": sets_bytes, "cases": compacted}


def main() -> None:
    import gc

    t0 = time.perf_counter()
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    from blast_radius import verify as V
    from blast_radius.db_cache import build_tpc_duckdb_files
    from blast_radius.import_state import import_all_sqlglot

    # Spawn DB cache recorder BEFORE importing tests in parent.
    pid_db = os.fork()
    if pid_db == 0:
        try:
            n_db = _record_db_cache_child()
            print(f"precompile_cases: recorded {n_db} DB cache entries", file=sys.stderr)
        finally:
            os._exit(0)

    import_all_sqlglot()
    V._SKIP_POOL_PICKLE = True
    V._patch_pool_references()
    src_prefix = str(SOURCE.resolve()) + "/sqlglot/"

    # Run the independent build-time recorders in parallel via fork() for ~3x faster builds.
    gc.freeze()
    opt_tmp = "/tmp/_br_pc_opt.marshal"
    val_tmp = "/tmp/_br_pc_val.marshal"

    def _spawn_recorder(tmp_path: str, fn) -> int:
        pid = os.fork()
        if pid == 0:
            mon = sys.monitoring
            tool = mon.COVERAGE_ID
            mon.use_tool_id(tool, "blast_radius_cases")
            try:
                data = fn(mon, tool)
                with open(tmp_path, "wb") as handle:
                    handle.write(marshal.dumps(data))
            finally:
                mon.set_events(tool, 0)
                mon.free_tool_id(tool)
            os._exit(0)
        return pid

    def _opt_and_blocks(mon, tool):
        base = {
            **_record_optimizer_cases(mon, tool, src_prefix),
            **_record_extra_subtest_cases(mon, tool, src_prefix),
        }
        return {**base, **_record_block_cases(mon, tool, src_prefix, skip_tids=set(base))}

    pid_opt = _spawn_recorder(opt_tmp, _opt_and_blocks)
    pid_val = _spawn_recorder(val_tmp, lambda mon, tool: _record_validator_cases(mon, tool, src_prefix))

    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "blast_radius_cases")
    try:
        tpc_cases, ckpts_payload = _record_tpc_and_ckpts(mon, tool, src_prefix)
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)

    build_tpc_duckdb_files(SOURCE)

    _, st_opt = os.waitpid(pid_opt, 0)
    _, st_val = os.waitpid(pid_val, 0)
    _, st_db = os.waitpid(pid_db, 0)
    if st_opt != 0 or st_val != 0 or st_db != 0:
        raise RuntimeError(f"precompile_cases child failed: opt={st_opt} val={st_val} db={st_db}")

    opt_cases = marshal.loads(open(opt_tmp, "rb").read())
    val_cases = marshal.loads(open(val_tmp, "rb").read())
    for p in (opt_tmp, val_tmp):
        try:
            os.unlink(p)
        except OSError:
            pass

    all_cases = {**opt_cases, **val_cases, **tpc_cases}
    CASES_FILE.write_bytes(marshal.dumps(_intern_cases(all_cases)))

    ckpt_meta = {}
    ckpt_bytes = 0
    for k, entry in ckpts_payload["checkpoints"].items():
        ckpt_bytes += _write_ckpt_bin(
            k,
            entry,
            ckpts_payload["expected"],
            ckpts_payload["schema"],
            ckpts_payload["sqls"] if k == 0 else None,
            for_exec=False,
        )
        ckpt_meta[k] = {
            "files": entry["files"],
            "suffix_files": entry["suffix_files"],
            "suffix_exec_files": entry["suffix_exec_files"],
            "est_ms": entry["est_ms"],
        }

    exec_ckpt_meta = {}
    for k, entry in ckpts_payload["exec_checkpoints"].items():
        ckpt_bytes += _write_ckpt_bin(
            k,
            entry,
            ckpts_payload["expected"],
            ckpts_payload["schema"],
            None,
            for_exec=True,
        )
        exec_ckpt_meta[k] = {
            "files": entry["files"],
            "suffix_exec_files": entry["suffix_exec_files"],
            "est_ms": entry["est_ms"],
        }

    meta_payload = {
        "checkpoints": ckpt_meta,
        "exec_checkpoints": exec_ckpt_meta,
        "stage_lines": ckpts_payload["stage_lines"],
        "s2b_query_ms": ckpts_payload["s2b_query_ms"],
        "rule_times_ms": ckpts_payload["rule_times_ms"],
        "exec_rule_times_ms": ckpts_payload["exec_rule_times_ms"],
        "sql_ms": ckpts_payload["sql_ms"],
        "gen_files": ckpts_payload["gen_files"],
        "parse_files": ckpts_payload["parse_files"],
        "s1_opt_files": ckpts_payload["s1_opt_files"],
        "s1_duck_files": ckpts_payload["s1_duck_files"],
        "s1_plan_files": ckpts_payload["s1_plan_files"],
        "s1_pygen_files": ckpts_payload["s1_pygen_files"],
        "max_nodes_tok_lines": ckpts_payload["max_nodes_tok_lines"],
    }
    with open(TPCDS_CKPTS_META_FILE, "wb") as handle:
        pickle.dump(meta_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    n_cf = sum(1 for v in opt_cases.values() if v["kind"] == "check_file")
    n_st = sum(1 for v in opt_cases.values() if v["kind"] == "subtest")
    n_blk = sum(1 for v in opt_cases.values() if v["kind"] == "blocks")
    print(
        f"case map in {time.perf_counter() - t0:.1f}s: "
        f"{n_cf} check_file, {n_st} subtest, {n_blk} blocks, {len(val_cases)} validator, {len(tpc_cases)} tpc; "
        f"cases={CASES_FILE.stat().st_size / 1e6:.2f} MB, "
        f"ckpt_meta={TPCDS_CKPTS_META_FILE.stat().st_size / 1e3:.1f} KB, "
        f"ckpts={ckpt_bytes / 1e6:.2f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
