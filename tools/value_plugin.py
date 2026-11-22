"""pytest plugin used only while building map/values (never at scoring time).

Rewrites every condition site in sqlglot (see blast_radius/valuesite.py) so that
evaluating it also records, for the running test, the truth of each and/or/not
operand and the outcome of every comparison relation on the compared values.
Atoms of side-effect-free sites that the program short-circuited are evaluated
afterwards (speculatively, with recording off) so a vector is complete.

Contexts: "<nodeid>|setup|run|teardown", "collect|<module>", "" (import / between tests).
Output: $MAP_OUT/values_<shard>.pkl = {"sites": {(file, key): nslots-width}, "rec": {ctx: {(file, key): set(vec)}}}.
"""

from __future__ import annotations

import ast
import concurrent.futures as cf
import importlib.abc
import importlib.machinery
import importlib.util
import operator
import os
import pickle
import sys

sys.path.insert(0, os.environ.get("BR_REPO", "/repo"))
from blast_radius import valuesite as vs  # noqa: E402

OUT = os.environ.get("MAP_OUT", "/out")
SHARD = os.environ.get("VALUE_SHARD", "0")
ROOT = os.getcwd()
PKG_DIR = os.path.join(ROOT, "sqlglot") + os.sep

_SPECS: list = []  # sid -> (spec, nslots, pure, file, key)
_rec: dict[str, dict[int, set[str]]] = {}
_cur: dict[int, set[str]] = _rec.setdefault("", {})
_spec_depth = 0
_MISSING = object()

_OPS = {
    "Is": operator.is_, "IsNot": operator.is_not, "Eq": operator.eq, "NotEq": operator.ne,
    "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
    "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
}
_ORDER = ("Lt", "LtE", "Gt", "GtE")


def _tc(v) -> str:
    if v is True:
        return "T"
    if v is False:
        return "F"
    try:
        return "t" if v else "f"
    except BaseException:
        return "E"


def _rel(fn, a, b) -> str:
    try:
        return _tc(fn(a, b))
    except BaseException:
        return "E"


def _sig(op: str, a, b, res) -> str:
    out = ["?"] * len(vs.RELS)
    own = _tc(res)
    out[vs.REL_INDEX[op]] = own
    if op in ("In", "NotIn"):
        other = "NotIn" if op == "In" else "In"
        out[vs.REL_INDEX[other]] = {"T": "F", "F": "T", "t": "F", "f": "T"}.get(own, "E")
        return "".join(out)
    is_ = a is b
    out[0], out[1] = ("T", "F") if is_ else ("F", "T")
    global _spec_depth
    _spec_depth += 1
    try:
        for name in ("Eq", "NotEq") + (_ORDER if op in _ORDER else ()):
            if name != op:
                out[vs.REL_INDEX[name]] = _rel(_OPS[name], a, b)
    finally:
        _spec_depth -= 1
    return "".join(out)


def _plain(s, fs, vals):
    kind = s[0]
    if kind == "A":
        v = vals[s[1]]
        if v is _MISSING:
            v = vals[s[1]] = fs[s[1]]()
        return v
    if kind == "N":
        return not _plain(s[1], fs, vals)
    if kind == "&":
        for c in s[1]:
            v = _plain(c, fs, vals)
            if not v:
                return v
        return v
    if kind == "|":
        for c in s[1]:
            v = _plain(c, fs, vals)
            if v:
                return v
        return v
    atoms, ops = s[1], s[2]
    left = fs[atoms[0]]()
    last = len(ops) - 1
    for j, op in enumerate(ops):
        right = fs[atoms[j + 1]]()
        res = _OPS[op](left, right)
        if j < last and not res:
            return res
        left = right
    return res


def _br_vp(sid, fs):
    if _spec_depth:
        return _plain(_SPECS[sid][0], fs, [_MISSING] * len(fs))
    spec, nslots, pure = _SPECS[sid][:3]
    rec = [None] * nslots
    vals = [_MISSING] * len(fs)

    def get(i):
        v = vals[i]
        if v is _MISSING:
            v = vals[i] = fs[i]()
        return v

    def ev(s):
        kind = s[0]
        if kind == "A":
            try:
                v = get(s[1])
            except BaseException:
                rec[s[2]] = "X"
                raise
            rec[s[2]] = _tc(v)
            return v
        if kind == "N":
            return not ev(s[1])
        if kind == "&":
            for c in s[1]:
                v = ev(c)
                if not v:
                    return v
            return v
        if kind == "|":
            for c in s[1]:
                v = ev(c)
                if v:
                    return v
            return v
        atoms, ops, slots = s[1], s[2], s[3]
        try:
            left = get(atoms[0])
        except BaseException:
            rec[slots[0]] = "X" * len(vs.RELS)
            raise
        last = len(ops) - 1
        for j, op in enumerate(ops):
            try:
                right = get(atoms[j + 1])
                res = _OPS[op](left, right)
            except BaseException:
                rec[slots[j]] = "X" * len(vs.RELS)
                raise
            rec[slots[j]] = _sig(op, left, right, res)
            if j < last and not res:
                return res
            left = right
        return res

    def finish():
        if _spec_depth:
            return
        parts = []
        for sl in range(nslots):
            parts.append(rec[sl] if rec[sl] is not None else _WIDTH_BLANK[_SPECS[sid][5][sl]])
        _cur.setdefault(sid, set()).add("".join(parts))

    try:
        result = ev(spec)
    except BaseException:
        finish()
        raise
    if pure and None in rec:
        _speculate(spec, rec, get)
    finish()
    return result


_WIDTH_BLANK = {1: "-", len(vs.RELS): "-" * len(vs.RELS)}


def _speculate(spec, rec, get) -> None:
    global _spec_depth
    _spec_depth += 1
    try:

        def walk(s):
            kind = s[0]
            if kind == "A":
                if rec[s[2]] is None:
                    try:
                        rec[s[2]] = _tc(get(s[1]))
                    except BaseException:
                        rec[s[2]] = "E"
            elif kind == "N":
                walk(s[1])
            elif kind in "&|":
                for c in s[1]:
                    walk(c)
            else:
                atoms, ops, slots = s[1], s[2], s[3]
                for j, op in enumerate(ops):
                    if rec[slots[j]] is not None:
                        continue
                    try:
                        a, b = get(atoms[j]), get(atoms[j + 1])
                        res = _OPS[op](a, b)
                    except BaseException:
                        rec[slots[j]] = "E" * len(vs.RELS)
                        continue
                    rec[slots[j]] = _sig(op, a, b, res)

        walk(spec)
    finally:
        _spec_depth -= 1


class _Rewriter(ast.NodeTransformer):
    def __init__(self, rel: str):
        self.rel = rel
        self.depth = 0
        self.count = 0

    def _scoped(self, node):
        self.depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _scoped

    def _wrap_expr(self, node: ast.expr, is_bare: bool = False):
        spec, atoms, nslots = vs.build(node)
        if not vs.instrumentable(atoms):
            return self.visit(node) if is_bare else super().visit(node)
        key = vs.site_key(node)
        pure = all(vs.is_pure(a) for a in atoms)
        widths = {}

        def wwalk(s):
            k = s[0]
            if k == "A":
                widths[s[2]] = 1
            elif k == "N":
                wwalk(s[1])
            elif k in "&|":
                for c in s[1]:
                    wwalk(c)
            else:
                for sl in s[3]:
                    widths[sl] = len(vs.RELS)

        wwalk(spec)
        sid = len(_SPECS)
        _SPECS.append((spec, nslots, pure, self.rel, key, widths))
        self.count += 1
        lambdas = []
        for atom in atoms:
            new_atom = self.visit(atom)
            lam = ast.Lambda(
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=new_atom,
            )
            lambdas.append(lam)
        call = ast.Call(
            func=ast.Name(id="_br_vp", ctx=ast.Load()),
            args=[ast.Constant(sid), ast.Tuple(elts=lambdas, ctx=ast.Load())],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def _visit_test_node(self, node):
        if self.depth > 0 and isinstance(node.test, ast.expr) and not vs.is_logic(node.test):
            node.test = self._wrap_expr(node.test, is_bare=True)
            for name, value in ast.iter_fields(node):
                if name == "test":
                    continue
                if isinstance(value, list):
                    node_list = [self.visit(item) if isinstance(item, ast.AST) else item for item in value]
                    setattr(node, name, node_list)
                elif isinstance(value, ast.AST):
                    setattr(node, name, self.visit(value))
            return node
        return self.generic_visit(node)

    visit_If = visit_While = visit_IfExp = _visit_test_node

    def visit(self, node):
        if self.depth > 0 and isinstance(node, ast.expr) and vs.is_logic(node):
            return self._wrap_expr(node, is_bare=False)
        return super().visit(node)


class _Loader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        tree = ast.parse(data, filename=path)
        tree = _Rewriter(rel).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True, optimize=_optimize)

    def exec_module(self, module):
        module.__dict__["_br_vp"] = _br_vp
        super().exec_module(module)

    def get_code(self, fullname):
        # Never use cached bytecode: always rewrite from source.
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            return spec
        if not os.path.abspath(spec.origin).startswith(PKG_DIR):
            return spec
        spec.loader = _Loader(fullname, spec.origin)
        return spec


sys.meta_path.insert(0, _Finder())


class _InProcessExecutor:
    """Stands in for ProcessPoolExecutor: runs each call now, through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future: cf.Future = cf.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _set_ctx(ctx: str) -> None:
    global _cur
    _cur = _rec.setdefault(ctx, {})


def _import_everything() -> None:
    import importlib
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name != "sqlglot.__main__":
            try:
                importlib.import_module(info.name)
            except Exception as err:
                print(f"value_plugin: failed to import {info.name}: {err}", file=sys.stderr)


import pytest  # noqa: E402


def pytest_configure(config):
    cf.ProcessPoolExecutor = _InProcessExecutor
    _set_ctx("")
    _import_everything()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        _set_ctx("collect|" + collector.nodeid)
        yield
        _set_ctx("")
    else:
        yield


def _phase(name):
    @pytest.hookimpl(hookwrapper=True)
    def hook(item, *args, **kwargs):
        _set_ctx(f"{item.nodeid}|{name}")
        yield
        _set_ctx("")

    return hook


pytest_runtest_setup = _phase("setup")
pytest_runtest_call = _phase("run")
pytest_runtest_teardown = _phase("teardown")

_outcomes: dict[str, str] = {}


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        _outcomes.setdefault(report.nodeid, report.outcome)


def pytest_unconfigure(config):
    sites = {sid: (s[3], s[4]) for sid, s in enumerate(_SPECS)}
    out = {
        "sites": sites,
        "rec": {ctx: {sid: sorted(v) for sid, v in d.items()} for ctx, d in _rec.items() if d},
        "outcomes": _outcomes,
        "n_sites": len(_SPECS),
    }
    with open(os.path.join(OUT, f"values_{SHARD}.pkl"), "wb") as handle:
        pickle.dump(out, handle, protocol=4)
    print(f"value_plugin: {len(_SPECS)} sites, {len(_rec)} contexts", file=sys.stderr)
