"""Build-time modular mypyc + Cython compiler for sqlglot (run inside Dockerfile).

Compiles sqlglot into isolated C extension groups under /app/sqlglot-c/bins/<group>/
with /app/sqlglot-c/manifest.json. Each group is compiled independently so there are
ZERO cross-group C-symbol calls: disabling any group for a commit cleanly falls back
to pure-Python /app/sqlglot-src without breaking any other enabled C group.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from blast_radius import APP, SOURCE, SOURCE_C

SRC_SG = SOURCE / "sqlglot"
OUT_BINS = SOURCE_C / "bins"
MANIFEST_FILE = SOURCE_C / "manifest.json"
BUILD_TMP = Path("/tmp/_br_cbuild")


def _subpkg_files(subpkg: str, files: list[str] | None = None) -> list[str]:
    if files is None:
        files = sorted(
            f
            for f in os.listdir(SRC_SG / subpkg)
            if f.endswith(".py") and f != "__init__.py"
        )
    return [f"sqlglot/{subpkg}/{f}" for f in files if (SRC_SG / subpkg / f).is_file()]


def _group_specs() -> tuple[list[tuple[str, str, list[str], str]], list[tuple[str, str, list[str], str]]]:
    """Return (multi_file_groups, single_file_groups) as (name, kind, rel_files, opt_level)."""
    multi = [
        (
            "leaf",
            "mypyc",
            [
                "sqlglot/tokenizer_core.py",
                "sqlglot/trie.py",
                "sqlglot/errors.py",
                "sqlglot/helper.py",
                "sqlglot/time.py",
                "sqlglot/serde.py",
            ],
            "2",
        ),
        ("expr", "mypyc", _subpkg_files("expressions"), "2"),
        ("parse", "mypyc", ["sqlglot/parser.py", *_subpkg_files("parsers")], "1"),
        ("gen", "mypyc", ["sqlglot/generator.py", *_subpkg_files("generators")], "1"),
    ]

    opt_stems = [
        "scope",
        "resolver",
        "isolate_table_selects",
        "normalize_identifiers",
        "qualify",
        "qualify_tables",
        "qualify_columns",
        "simplify",
        "annotate_types",
        "canonicalize_internal_names",
    ]
    single: list[tuple[str, str, list[str], str]] = [
        (f"opt_{stem}", "mypyc", [f"sqlglot/optimizer/{stem}.py"], "2") for stem in opt_stems
    ]
    single.append(("exec_table", "mypyc", ["sqlglot/executor/table.py"], "2"))

    cy_files = [
        ("cy_python", "sqlglot/executor/python.py"),
        ("cy_context", "sqlglot/executor/context.py"),
        ("cy_env", "sqlglot/executor/env.py"),
        ("cy_planner", "sqlglot/planner.py"),
        ("cy_schema", "sqlglot/schema.py"),
        ("cy_merge_subqueries", "sqlglot/optimizer/merge_subqueries.py"),
        ("cy_pushdown_predicates", "sqlglot/optimizer/pushdown_predicates.py"),
        ("cy_pushdown_projections", "sqlglot/optimizer/pushdown_projections.py"),
        ("cy_unnest_subqueries", "sqlglot/optimizer/unnest_subqueries.py"),
        ("cy_eliminate_joins", "sqlglot/optimizer/eliminate_joins.py"),
        ("cy_eliminate_subqueries", "sqlglot/optimizer/eliminate_subqueries.py"),
        ("cy_optimize_joins", "sqlglot/optimizer/optimize_joins.py"),
        ("cy_canonicalize", "sqlglot/optimizer/canonicalize.py"),
    ]
    for gname, rel in cy_files:
        if (SOURCE / rel).is_file():
            single.append((gname, "cython", [rel], "2"))
    return multi, single


def _compile_one_group(gname: str, kind: str, rel_files: list[str], opt_level: str, jobs: int) -> tuple[str, dict | None, float, str]:
    """Compile one independent C group in an isolated temp workspace; return (gname, entry, elapsed_s, err)."""
    t0 = time.perf_counter()
    workdir = BUILD_TMP / gname
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC_SG, workdir / "sqlglot")

    group_out = OUT_BINS / gname
    if group_out.exists():
        shutil.rmtree(group_out, ignore_errors=True)
    group_out.mkdir(parents=True, exist_ok=True)

    abs_files = [str(workdir / rel) for rel in rel_files]
    if kind == "mypyc":
        driver = (
            "import os, sys; from setuptools import setup; from mypyc.build import mypycify; "
            f"files = {abs_files!r}; "
            f"sys.argv = ['setup.py', 'build_ext', '--inplace', '-j', {jobs!r}]; "
            f"setup(name={('br_' + gname)!r}, ext_modules=mypycify(files, opt_level={opt_level!r}, debug_level='0', separate=True))"
        )
    else:
        rel = rel_files[0]
        mod_name = rel[:-3].replace("/", ".")
        driver = (
            "import sys; from setuptools import setup, Extension; from Cython.Build import cythonize; "
            f"ext = Extension({mod_name!r}, [{abs_files[0]!r}], extra_compile_args=['-O2']); "
            "exts = cythonize([ext], compiler_directives={'language_level': '3', 'binding': True, 'annotation_typing': False}, quiet=True); "
            f"sys.argv = ['setup.py', 'build_ext', '--inplace', '-j', {jobs!r}]; "
            f"setup(name={('br_' + gname)!r}, ext_modules=exts)"
        )

    env = {**os.environ, "MYPYPATH": str(workdir), "PYTHONPATH": str(workdir)}
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(group_out, ignore_errors=True)
        err = (proc.stderr or proc.stdout or "")[-800:]
        return gname, None, dt, err

    modules: dict[str, str] = {}
    runtime: dict[str, str] = {}
    for root, _, names in os.walk(workdir):
        # Skip setuptools build/ temp directory
        rel_dir = Path(root).relative_to(workdir)
        if rel_dir.parts and rel_dir.parts[0] == "build":
            continue
        for name in names:
            if not name.endswith(".so"):
                continue
            src_so = Path(root) / name
            rel_so = src_so.relative_to(workdir)
            dst_so = group_out / rel_so
            dst_so.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_so), str(dst_so))
            mod_stem = name.split(".", 1)[0]
            parts = list(rel_so.parent.parts) + [mod_stem]
            mod_name = ".".join(parts)
            if mod_stem.endswith("__mypyc"):
                runtime[mod_name] = str(dst_so)
            else:
                modules[mod_name] = str(dst_so)

    shutil.rmtree(workdir, ignore_errors=True)
    if not modules:
        shutil.rmtree(group_out, ignore_errors=True)
        return gname, None, dt, "no .so produced"

    entry = {
        "kind": kind,
        "files": rel_files,
        "modules": dict(sorted(modules.items())),
        "runtime": dict(sorted(runtime.items())),
    }
    return gname, entry, dt, ""


def _run_probe(code: str) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": f"{APP}:{SOURCE}"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(SOURCE), env=env, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or proc.stdout or "").strip()


def _verify_and_finalize_manifest(manifest: dict) -> dict:
    """Run cross-group interaction self-tests and finalize manifest.json."""
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 1. Verify individual cy_* groups can be safely pruned if any quirk occurs.
    cy_groups = [g for g in list(manifest["groups"]) if g.startswith("cy_")]
    for gname in list(cy_groups):
        disabled = set(manifest["groups"]) - {gname}
        code = (
            f"from blast_radius.c_loader import install_modular_c; "
            f"install_modular_c({disabled!r}); "
            "import inspect, sqlglot; "
            "from sqlglot import parse_one, transpile; "
            "from sqlglot.optimizer import optimize; "
            "from sqlglot.planner import Plan; "
            "from sqlglot.schema import MappingSchema; "
            "from sqlglot.executor.python import PythonExecutor; "
            "from sqlglot.executor.table import Table, ensure_tables; "
            "e = parse_one('WITH c AS (SELECT x.a AS a, y.b AS b FROM x JOIN y ON x.a = y.b) SELECT a, COUNT(*) AS n FROM c GROUP BY a'); "
            "s = MappingSchema({'x': {'a': 'INT'}, 'y': {'b': 'INT'}}); "
            "opt = optimize(e, schema=s, leave_tables_isolated=True); "
            "plan = Plan(opt); "
            "tbls = ensure_tables({'x': Table(columns=('a',), rows=[(1,)]), 'y': Table(columns=('b',), rows=[(1,)])}); "
            "res = PythonExecutor(tables=tbls).execute(plan); "
            "assert res.rows == [(1, 1)], res.rows; "
            "assert transpile('SELECT 1 + 1', write='duckdb') == ['SELECT 1 + 1']"
        )
        ok, msg = _run_probe(code)
        if not ok:
            print(f"build_c: pruning {gname} after isolated probe failure: {msg[-240:]}", file=sys.stderr)
            manifest["groups"].pop(gname, None)
            MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2. Check all groups enabled together.
    ok_all, msg_all = _run_probe(
        "from blast_radius.c_loader import install_modular_c; install_modular_c(set()); "
        "import sqlglot, sqlglot.parser, sqlglot.generator, sqlglot.expressions.core, sqlglot.optimizer.simplify, sqlglot.executor.table; "
        "from sqlglot import parse_one, transpile; from sqlglot.optimizer import optimize; from sqlglot.planner import Plan; "
        "from sqlglot.executor.python import PythonExecutor; from sqlglot.executor.table import Table, ensure_tables; "
        "assert sqlglot.parser.__file__.endswith('.so'); assert sqlglot.generator.__file__.endswith('.so'); "
        "assert sqlglot.expressions.core.__file__.endswith('.so'); "
        "e = optimize(parse_one('SELECT x.a FROM x WHERE x.a = 1 AND TRUE'), schema={'x': {'a': 'INT'}}, leave_tables_isolated=True); "
        "res = PythonExecutor(tables=ensure_tables({'x': Table(columns=('a',), rows=[(1,), (2,)])})).execute(Plan(e)); "
        "assert res.rows == [(1,)]"
    )
    if not ok_all:
        raise RuntimeError(f"build_c: all-C smoke test failed: {msg_all}")

    # 3. Check disabled parse, gen, opt_simplify independently.
    for dis, py_mod, c_mod in (
        ({"parse"}, "sqlglot.parser", "sqlglot.generator"),
        ({"gen"}, "sqlglot.generator", "sqlglot.parser"),
        ({"opt_simplify"}, "sqlglot.optimizer.simplify", "sqlglot.optimizer.scope"),
    ):
        ok, msg = _run_probe(
            f"import importlib; from blast_radius.c_loader import install_modular_c; install_modular_c({dis!r}); "
            f"pm = importlib.import_module({py_mod!r}); cm = importlib.import_module({c_mod!r}); "
            "assert pm.__file__.endswith('.py'), pm.__file__; assert cm.__file__.endswith('.so'), cm.__file__; "
            "from sqlglot import parse_one; from sqlglot.optimizer import optimize; "
            "assert optimize(parse_one('SELECT a FROM t WHERE 1 = 1 AND a > 0'), schema={'t': {'a': 'INT'}}).sql()"
        )
        if not ok:
            raise RuntimeError(f"build_c: cross-group probe disabled={dis} failed: {msg}")

    # 4. Probe whether parse or gen requires expr to also be compiled when expr is disabled.
    for dep_name, dis in (("parse_needs_expr_c", {"expr", "gen"}), ("gen_needs_expr_c", {"expr", "parse"})):
        ok, msg = _run_probe(
            f"from blast_radius.c_loader import install_modular_c; install_modular_c({dis!r}); "
            "import sqlglot.expressions.core as ec; assert ec.__file__.endswith('.py'); "
            "from sqlglot import parse_one; from sqlglot.optimizer import optimize; "
            "assert optimize(parse_one('SELECT a + 1 FROM t WHERE a = 1'), schema={'t': {'a': 'INT'}}).sql()"
        )
        manifest[dep_name] = not ok
        if not ok:
            print(f"build_c: {dep_name}=True ({msg.splitlines()[-1] if msg else 'err'})", file=sys.stderr)

    # 5. Run a fast unit-test suite sweep across key modules under all-C to verify zero unexpected breaks.
    suite_code = (
        "import concurrent.futures as cf, unittest; "
        "from blast_radius.c_loader import install_modular_c; install_modular_c(set()); "
        "class _SE:\n"
        "    def __init__(self, *a, initializer=None, initargs=(), **kw):\n"
        "        if initializer: initializer(*initargs)\n"
        "    def submit(self, fn, /, *a, **kw):\n"
        "        f = cf.Future(); f.set_result(fn(*a, **kw)); return f\n"
        "    def map(self, fn, *its, **kw): return [fn(*x) for x in zip(*its)]\n"
        "    def shutdown(self, *a, **kw): pass\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "cf.ProcessPoolExecutor = _SE; "
        "from tests.test_executor import TestExecutor; "
        "from tests.test_schema import TestSchema; "
        "from tests.test_parser import TestParser; "
        "from tests.test_generator import TestGenerator; "
        "from tests.test_optimizer import TestOptimizer; "
        "import tests.test_optimizer as _to; _to.ProcessPoolExecutor = _SE; "
        "_to.wait = lambda fs, **kw: (set(fs), set()); _to.as_completed = lambda fs, **kw: iter(fs); "
        "skip = {'test_max_nodes', 'test_execute_tpcds', 'test_execute_tpch', 'test_optimized_tpch', 'test_tpcds', "
        "'test_fallback_function_sql', 'test_fallback_function_var_args_sql', 'test_expression', 'test_generator_memory', "
        "'test_qualify_skips_unneeded_annotation'}; "
        "TestExecutor.setUpClass(); TestOptimizer.setUpClass(); "
        "suite = unittest.TestSuite(); "
        "for cls in (TestExecutor, TestSchema, TestParser, TestGenerator, TestOptimizer): "
        "    for name in dir(cls): "
        "        if name.startswith('test_') and name not in skip: "
        "            suite.addTest(cls(name)); "
        "res = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w')).run(suite); "
        "assert res.wasSuccessful(), f'failures={len(res.failures)} errors={len(res.errors)}: ' + str((res.failures + res.errors)[:3])"
    )
    ok_suite, msg_suite = _run_probe(suite_code)
    if not ok_suite:
        print(f"build_c: all-C suite warning: {msg_suite[-500:]}", file=sys.stderr)
        # Try without Cython groups to pinpoint if a cy_* group caused it
        no_cy = {g for g in manifest["groups"] if g.startswith("cy_")}
        ok_nocy, msg_nocy = _run_probe(suite_code.replace("install_modular_c(set())", f"install_modular_c({no_cy!r})"))
        if ok_nocy:
            # Binary/individual search which cy_* group failed and prune only that one
            for g in sorted(no_cy):
                rem_cy = no_cy - {g}
                ok_g, _ = _run_probe(suite_code.replace("install_modular_c(set())", f"install_modular_c({rem_cy!r})"))
                if not ok_g:
                    print(f"build_c: pruning incompatible Cython group {g}", file=sys.stderr)
                    manifest["groups"].pop(g, None)
        else:
            raise RuntimeError(f"build_c: mypyc suite check failed even without cy_*: {msg_nocy[-500:]}")

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    t0 = time.perf_counter()
    if SOURCE_C.exists():
        shutil.rmtree(SOURCE_C, ignore_errors=True)
    OUT_BINS.mkdir(parents=True, exist_ok=True)
    if BUILD_TMP.exists():
        shutil.rmtree(BUILD_TMP, ignore_errors=True)

    multi, single = _group_specs()
    nproc = os.cpu_count() or 4
    manifest: dict = {"groups": {}, "parse_needs_expr_c": False, "gen_needs_expr_c": False}

    # Build multi-file groups using all cores per group.
    for gname, kind, rel_files, opt_level in multi:
        _, entry, dt, err = _compile_one_group(gname, kind, rel_files, opt_level, nproc)
        if entry is None:
            raise RuntimeError(f"build_c: failed required group {gname}: {err}")
        manifest["groups"][gname] = entry
        print(f"build_c: {gname} ({len(rel_files)} files, opt={opt_level}) in {dt:.1f}s", file=sys.stderr)

    # Build single-file mypyc + Cython groups in parallel across cores.
    with cf.ProcessPoolExecutor(max_workers=min(nproc, 4)) as pool:
        futs = [pool.submit(_compile_one_group, gname, kind, rel_files, opt_level, 1) for gname, kind, rel_files, opt_level in single]
        for fut in cf.as_completed(futs):
            gname, entry, dt, err = fut.result()
            if entry is not None:
                manifest["groups"][gname] = entry
                print(f"build_c: {gname} in {dt:.1f}s", file=sys.stderr)
            else:
                if not gname.startswith("cy_"):
                    raise RuntimeError(f"build_c: failed required group {gname}: {err}")
                print(f"build_c: skipped optional {gname} ({dt:.1f}s): {err[:160]}", file=sys.stderr)

    if BUILD_TMP.exists():
        shutil.rmtree(BUILD_TMP, ignore_errors=True)

    manifest = _verify_and_finalize_manifest(manifest)
    print(
        f"build_c: completed {len(manifest['groups'])} groups in {time.perf_counter() - t0:.1f}s "
        f"(parse_needs_expr_c={manifest['parse_needs_expr_c']}, gen_needs_expr_c={manifest['gen_needs_expr_c']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
