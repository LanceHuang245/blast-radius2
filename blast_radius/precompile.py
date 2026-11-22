"""Build-time precompiler run inside Dockerfile (never at scoring time).

Reads map/impact.json.gz + /app/sqlglot-src, recomputes clean scoped Expression
class_refs, strips false-positive EXPRESSION_METADATA import_reads, maps
sqlglot/typing/*.py modules to Dialect classes, profiles real single-process
CPU times for heavy tests, appends test_optimized_tpch baseline SQL pairs to
tpc_meta.pkl, and writes /app/map/impact.marshal.
"""

from __future__ import annotations

import ast
import concurrent.futures as cf
import gc
import gzip
import importlib
import json
import marshal
import os
import pickle
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

from blast_radius import MAP_FILE, SOURCE
from blast_radius.impact_map import module_name
from blast_radius.precompile_tpc import TPC_META_FILE

# Non-TPC / non-max_nodes tests with recorded duration >= 200ms in pinned sqlglot.
CALIBRATED_TESTS = (
    "tests/test_optimizer.py::TestOptimizer::test_scope",
    "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "tests/test_optimizer.py::TestOptimizer::test_annotate_funcs",
    "tests/test_optimizer.py::TestOptimizer::test_annotate_types",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_columns",
    "tests/test_optimizer.py::TestOptimizer::test_eliminate_joins",
    "tests/test_optimizer.py::TestOptimizer::test_eliminate_subqueries",
    "tests/test_optimizer.py::TestOptimizer::test_pushdown_predicates",
    "tests/test_optimizer.py::TestOptimizer::test_pushdown_projections",
    "tests/test_executor.py::TestExecutor::test_agg_order",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_in_nulls",
    "tests/test_schema.py::TestSchema::test_cache_does_not_retain_queries",
    "tests/test_optimizer.py::TestOptimizer::test_merge_subqueries",
    "tests/test_optimizer.py::TestOptimizer::test_canonicalize",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_struct_star_long_cte_chain",
    "tests/test_serde.py::TestSerde::test_recursion",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_correlated_all",
    "tests/dialects/test_snowflake.py::TestSnowflake::test_snowflake",
    "tests/test_optimizer.py::TestOptimizer::test_simplify",
    "tests/test_serde.py::TestSerde::test_serde",
    "tests/dialects/test_redshift.py::TestRedshift::test_values",
    "tests/test_optimizer.py::TestOptimizer::test_deep_ast_type_annotation",
    "tests/dialects/test_bigquery.py::TestBigQuery::test_bigquery",
    "tests/test_optimize_cross_joins.py::TestOptimizeCrossJoins::test_conditional_cross_joins",
    "tests/test_optimizer.py::TestOptimizer::test_optimize",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_semi_anti_join_alias_shadowing_column",
    "tests/test_optimizer.py::TestOptimizer::test_tpch",
    "tests/test_expressions.py::TestExprs::test_is_star",
    "tests/test_optimizer.py::TestOptimizer::test_unnest_subqueries",
    "tests/test_transpile.py::TestTranspile::test_identity",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_scalar_subquery_predicates",
    "tests/test_expressions.py::TestExprs::test_named_selects_long_by_name_chain",
    "tests/dialects/test_duckdb.py::TestDuckDB::test_duckdb",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_uncorrelated_exists",
    "tests/dialects/test_clickhouse.py::TestClickhouse::test_clickhouse",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_special_joins",
    "tests/test_optimizer.py::TestOptimizer::test_canonicalize_internal_names",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_columns_using_after_outer_join",
)


def _parse_trees(src: Path) -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for top in ("sqlglot", "tests"):
        top_dir = src / top
        if not top_dir.is_dir():
            continue
        for root, _, names in os.walk(top_dir):
            for name in names:
                if name.endswith(".py"):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, src).replace(os.sep, "/")
                    try:
                        trees[rel] = ast.parse(open(full, encoding="utf-8").read())
                    except SyntaxError:
                        continue
    return trees


def _scoped_class_refs(
    trees: dict[str, ast.Module],
    lines_idx: dict[str, dict[str, int]],
    masks: list[int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute Expression class_refs (`exp.<Cls>.<member>`) and all runtime `exp.<Cls>` references."""
    defs_by_name: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/expressions/"):
            continue
        mod = module_name(rel)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                defs_by_name[node.name].append((f"{mod}:{node.name}", rel, node.lineno, node.end_lineno))

    refs: dict[str, int] = defaultdict(int)
    any_refs: dict[str, int] = defaultdict(int)
    for rel, tree in trees.items():
        per_file = lines_idx.get(rel)
        if not per_file:
            continue

        imported_exp: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (
                (node.module and (node.module == "sqlglot.expressions" or node.module.startswith("sqlglot.expressions.")))
                or (rel.startswith("sqlglot/expressions/") and node.level >= 1)
            ):
                for al in node.names:
                    if al.name in defs_by_name:
                        imported_exp[al.asname or al.name] = al.name
        if rel.startswith("sqlglot/expressions/"):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in defs_by_name:
                    imported_exp[node.name] = node.name

        def record(cls_name: str, ln: int | None, target: dict[str, int], by_name: bool = False) -> None:
            idx = per_file.get(str(ln)) if ln is not None else None
            if idx is None:
                return
            mask = masks[idx]
            for ckey, def_rel, start, end in defs_by_name[cls_name]:
                if rel == def_rel and start <= ln <= end:
                    continue
                target[cls_name if by_name else ckey] |= mask

        def walk(node: ast.AST) -> None:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "exp"
                    and node.value.attr in defs_by_name
                ):
                    record(node.value.attr, getattr(node, "lineno", None), refs)
                    record(node.value.attr, getattr(node, "lineno", None), any_refs, by_name=True)
                elif isinstance(node.value, ast.Name) and node.value.id == "exp" and node.attr in defs_by_name:
                    record(node.attr, getattr(node, "lineno", None), any_refs, by_name=True)
                elif isinstance(node.value, ast.Name) and node.value.id in imported_exp:
                    record(imported_exp[node.value.id], getattr(node, "lineno", None), refs)
                    record(imported_exp[node.value.id], getattr(node, "lineno", None), any_refs, by_name=True)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported_exp:
                record(imported_exp[node.id], getattr(node, "lineno", None), any_refs, by_name=True)
            if isinstance(node, ast.arg):
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    walk(dec)
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    walk(d)
                for stmt in node.body:
                    walk(stmt)
                return
            if isinstance(node, ast.AnnAssign):
                walk(node.target)
                if node.value is not None:
                    walk(node.value)
                return
            for ch in ast.iter_child_nodes(node):
                walk(ch)

        walk(tree)
    return refs, any_refs


def _typing_dialect_map(src: Path, trees: dict[str, ast.Module]) -> dict[str, list[str]]:
    """Map each sqlglot/typing/*.py file to the Dialect class keys that consume its EXPRESSION_METADATA."""
    typing_parent: dict[str, str] = {}
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/typing/"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sqlglot.typing"):
                if any(a.name == "EXPRESSION_METADATA" for a in node.names):
                    p_rel = node.module.replace(".", "/") + ".py"
                    if not (src / p_rel).is_file():
                        p_rel = node.module.replace(".", "/") + "/__init__.py"
                    typing_parent[rel] = p_rel

    dialect_for_typing: dict[str, str] = {}
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/dialects/"):
            continue
        t_mod = cls_key = None
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sqlglot.typing"):
                if any(a.name == "EXPRESSION_METADATA" for a in node.names):
                    t_mod = node.module.replace(".", "/") + ".py"
                    if not (src / t_mod).is_file():
                        t_mod = node.module.replace(".", "/") + "/__init__.py"
            elif isinstance(node, ast.ClassDef):
                for s in node.body:
                    if isinstance(s, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "EXPRESSION_METADATA" for t in s.targets
                    ):
                        cls_key = f"{module_name(rel)}:{node.name}"
        if t_mod and cls_key:
            dialect_for_typing[t_mod] = cls_key

    out: dict[str, list[str]] = {}
    for t_file in sorted(dialect_for_typing):
        desc: list[str] = []
        for k in sorted(dialect_for_typing):
            cur: str | None = k
            while cur:
                if cur == t_file:
                    desc.append(dialect_for_typing[k])
                    break
                cur = typing_parent.get(cur)
        out[t_file] = desc
    return out


def _calibrate_and_update_tpc_meta(
    src: Path,
    all_test_ids: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Measure pure-Python and modular-C single-process CPU of heavy tests and append opt_tpch_pairs.

    Runs in clean forked children BEFORE parent imports sqlglot so C vs pure-Python measurements
    do not share sys.modules state. Both children use db_cache + gc.freeze() + high GC threshold
    matching Phase-2 scoring children.
    """
    from blast_radius.c_loader import PURE_PYTHON_ONLY_TESTS, load_c_manifest

    class _SyncExec:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
            if initializer is not None:
                initializer(*initargs)

        def submit(self, fn, /, *args, **kwargs):
            fut: cf.Future = cf.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as err:
                fut.set_exception(err)
            return fut

        def map(self, fn, *iterables, **kwargs):
            return [self.submit(fn, *a).result() for a in zip(*iterables)]

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    skip_ids = frozenset({
        "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
        "tests/test_parser.py::TestParser::test_max_nodes",
        "tests/test_executor.py::TestExecutor::test_execute_tpcds",
        "tests/test_executor.py::TestExecutor::test_execute_tpch",
        "tests/test_executor.py::TestExecutor::test_optimized_tpch",
    })
    to_calibrate: list[str] = list(CALIBRATED_TESTS)
    seen_cal = set(to_calibrate)
    for tid in all_test_ids:
        if tid in seen_cal or tid in skip_ids:
            continue
        if tid.count("::") == 2 and not tid.startswith("tests/test_dialect_imports.py::"):
            to_calibrate.append(tid)
            seen_cal.add(tid)

    def _run_child(use_c: bool, out_path: str) -> None:
        os.chdir(src)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        cf.ProcessPoolExecutor = _SyncExec
        if use_c:
            from blast_radius.c_loader import install_modular_c

            install_modular_c(set(), after_overlay=False)
        from blast_radius.db_cache import install_db_cache, wrap_duckdb_if_loaded

        install_db_cache()

        # Pure-Python child updates tpc_meta["opt_tpch_pairs"] cleanly if not yet present.
        if not use_c and TPC_META_FILE.is_file():
            with open(TPC_META_FILE, "rb") as handle:
                tpc_meta = pickle.load(handle)
            if "opt_tpch_pairs" not in tpc_meta:
                from sqlglot import transpile
                from tests.helpers import load_sql_fixture_pairs

                tpch_sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
                tpc_meta["opt_tpch_pairs"] = [
                    (
                        sql,
                        optimized,
                        transpile(sql, write="duckdb")[0],
                        transpile(optimized, write="duckdb")[0],
                    )
                    for _, sql, optimized in tpch_sqls
                ]
                with open(TPC_META_FILE, "wb") as handle:
                    pickle.dump(tpc_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        real_cpu_ms: dict[str, int] = {}
        loaded: dict[tuple[str, str], type | None] = {}
        for tid in to_calibrate:
            if use_c and tid in PURE_PYTHON_ONLY_TESTS:
                continue
            file_path, cls_name, meth_name = tid.split("::")
            mod_name = file_path[:-3].replace("/", ".")
            key = (mod_name, cls_name)
            if key not in loaded:
                try:
                    mod = importlib.import_module(mod_name)
                    wrap_duckdb_if_loaded()
                    if hasattr(mod, "ProcessPoolExecutor"):
                        setattr(mod, "ProcessPoolExecutor", _SyncExec)
                    cls = getattr(mod, cls_name)
                    if cls_name != "TestExecutor":
                        cls.setUpClass()
                    loaded[key] = cls
                except Exception:
                    loaded[key] = None
            cls = loaded[key]
            if cls is None:
                continue
            try:
                gc.collect()
                gc.freeze()
                gc.set_threshold(1_000_000, 50, 100)
                t0 = time.process_time()
                res = unittest.TestResult()
                cls(meth_name).run(res)
                dt_ms = max(1, round((time.process_time() - t0) * 1000))
                if res.wasSuccessful():
                    real_cpu_ms[tid] = dt_ms
            except Exception:
                continue

        tpc_c_ms: dict[str, int] = {}
        if use_c:
            try:
                from sqlglot import parse_one
                from sqlglot.optimizer import optimize
                from sqlglot.schema import ensure_schema
                from blast_radius.tpc_verify import load_tpcds_ckpt_stage, verify_tpc_test

                dummy_patch = {"sqlglot/optimizer/optimizer.py": ""}
                t0 = time.process_time()
                if verify_tpc_test("tpcds", "fast_a", patched_files=dummy_patch):
                    tpc_c_ms["exec_fast_a"] = max(50, round((time.process_time() - t0) * 1000))

                s0 = load_tpcds_ckpt_stage(0, for_exec=False)
                if s0 and s0.get("sqls"):
                    sch = ensure_schema(s0["schema"])
                    t0 = time.process_time()
                    for raw_sql in s0["sqls"]:
                        if raw_sql:
                            optimize(parse_one(raw_sql), schema=sch).sql(pretty=True)
                    tpc_c_ms["opt_tpcds_full"] = max(50, round((time.process_time() - t0) * 1000))
            except Exception:
                pass

        with open(out_path, "wb") as handle:
            marshal.dump({"cpu": real_cpu_ms, "tpc": tpc_c_ms}, handle)

    py_tmp = "/tmp/_br_cal_py.marshal"
    c_tmp = "/tmp/_br_cal_c.marshal"
    has_c = load_c_manifest() is not None

    def _spawn(use_c: bool, out_path: str) -> int:
        pid = os.fork()
        if pid == 0:
            try:
                _run_child(use_c, out_path)
            finally:
                os._exit(0)
        return pid

    # Run Py child first (it updates tpc_meta.pkl if needed), alongside C child in parallel!
    pid_py = _spawn(False, py_tmp)
    pid_c = _spawn(True, c_tmp) if has_c else None
    _, st_py = os.waitpid(pid_py, 0)
    st_c = os.waitpid(pid_c, 0)[1] if pid_c is not None else 0
    if st_py != 0 or st_c != 0:
        raise RuntimeError(f"precompile calibration child failed: py={st_py} c={st_c}")

    py_data = marshal.loads(open(py_tmp, "rb").read())
    c_data = marshal.loads(open(c_tmp, "rb").read()) if (has_c and os.path.isfile(c_tmp)) else {"cpu": {}, "tpc": {}}
    for p in (py_tmp, c_tmp):
        try:
            os.unlink(p)
        except OSError:
            pass
    return py_data["cpu"], c_data["cpu"], c_data["tpc"]


def _leaf_sqlglot_classes() -> list[str]:
    """Return sqlglot Dialect/Tokenizer/Parser/Generator classes with zero sqlglot subclasses."""
    import pkgutil

    os.chdir(SOURCE)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import sqlglot
    from sqlglot import generator, parser, tokens
    from sqlglot.dialects import dialect

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name != "sqlglot.__main__":
            try:
                importlib.import_module(info.name)
            except Exception:
                pass

    seen: set[type] = set()
    stack: list[type] = [dialect.Dialect, tokens.Tokenizer, parser.Parser, generator.Generator]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return sorted(
        {
            f"{c.__module__}:{c.__qualname__}"
            for c in seen
            if c.__module__.startswith("sqlglot")
            and not any(s.__module__.startswith("sqlglot") for s in c.__subclasses__())
        }
    )


def main() -> None:
    t0 = time.perf_counter()
    with gzip.open(MAP_FILE, "rt") as handle:
        raw = json.load(handle)

    old_masks = [int(s, 16) for s in raw["sets"]]
    trees = _parse_trees(SOURCE)
    pinned_ids: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                pinned_ids.add(node.id)
            elif isinstance(node, ast.Attribute):
                pinned_ids.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pinned_ids.add(node.name)
    scoped_refs, exp_any_refs = _scoped_class_refs(trees, raw["lines"], old_masks)
    exp_usage: dict[str, int] = defaultdict(int, exp_any_refs)
    for ckey, idx in raw["classes"].items():
        if ckey.startswith("sqlglot.expressions."):
            cname = ckey.rsplit(":", 1)[-1]
            if "." not in cname and not cname.startswith("_"):
                exp_usage[cname] |= old_masks[idx]
    typing_dialects = _typing_dialect_map(SOURCE, trees)

    # Strip false-positive `|EXPRESSION_METADATA` from import_reads/import_value_attrs (caused by class-body `.copy()`).
    raw["import_reads"] = {
        k: v for k, v in raw.get("import_reads", {}).items() if not k.endswith("|EXPRESSION_METADATA")
    }
    raw["import_any_attrs"] = [a for a in raw.get("import_any_attrs", []) if a != "EXPRESSION_METADATA"]
    raw["import_value_attrs"] = [a for a in raw.get("import_value_attrs", []) if a != "EXPRESSION_METADATA"]
    raw["typing_dialects"] = typing_dialects
    raw["pinned_identifiers"] = sorted(pinned_ids)
    cal_py, cal_c, tpc_c_ms = _calibrate_and_update_tpc_meta(SOURCE, raw["tests"])
    raw["leaf_classes"] = _leaf_sqlglot_classes()
    cal_py.setdefault("tests/test_generator.py::TestGenerator::test_generator_memory", 500)
    cal_py["tests/dialects/test_dialect.py::TestDialect::test_lazy_load"] = 5
    raw["real_cpu_ms"] = cal_py
    raw["real_cpu_ms_py"] = cal_py
    raw["real_cpu_ms_c"] = cal_c
    raw["tpc_c_ms"] = tpc_c_ms

    dict_any: dict[str, int] = defaultdict(int)
    for k, idx in raw.get("dict_keys", {}).items():
        dict_any[k.rsplit("|", 1)[0]] |= old_masks[idx]

    seen: dict[int, int] = {m: i for i, m in enumerate(old_masks)}
    sets_bytes: list[bytes] = [
        m.to_bytes((m.bit_length() + 7) // 8, "little") for m in old_masks
    ]

    def intern(mask: int) -> int:
        idx = seen.get(mask)
        if idx is None:
            idx = len(sets_bytes)
            seen[mask] = idx
            sets_bytes.append(mask.to_bytes((mask.bit_length() + 7) // 8, "little"))
        return idx

    raw["class_refs"] = {c: intern(m) for c, m in sorted(scoped_refs.items()) if m}
    raw["exp_usage"] = {c: intern(m) for c, m in sorted(exp_usage.items())}
    raw["dict_usage"] = {k: intern(m) for k, m in sorted(dict_any.items()) if m}
    raw["sets"] = sets_bytes
    raw["sets_binary"] = True

    out_path = MAP_FILE.parent / "impact.marshal"
    with open(out_path, "wb") as handle:
        marshal.dump(raw, handle)
    print(
        f"precompiled {out_path} in {time.perf_counter() - t0:.1f}s: "
        f"{len(raw['class_refs'])} scoped class_refs, "
        f"{len(raw['typing_dialects'])} typing dialects, "
        f"{len(raw['real_cpu_ms_py'])} py / {len(raw['real_cpu_ms_c'])} c calibrated tests, "
        f"tpc_c={tpc_c_ms}, "
        f"{len(raw['leaf_classes'])} leaf classes, "
        f"{out_path.stat().st_size / 1e6:.2f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
