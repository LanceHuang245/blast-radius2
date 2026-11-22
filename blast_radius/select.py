"""Per-file and per-change-block test selection rules."""

from __future__ import annotations

import ast
import marshal
import os
import tomllib
from dataclasses import dataclass, field

from blast_radius import APP, SOURCE, Everything, log
from blast_radius.impact_map import ImpactMap, module_name
from blast_radius.import_state import STATE_FILE
from blast_radius.patches import FilePatch, apply, change_blocks, parse_patch
from blast_radius.scopes import (
    COMPOUND_NODES,
    Scopes,
    _ast_dict_key_tokens,
    _extract_dict_entries,
    _is_docstring,
    canon_ast,
    comment_lines_only,
    header_end,
    indent,
    same_ast,
    same_code,
    same_local_binding_semantics,
)

SAFE_HOTPATCH_DECORATORS = frozenset({"staticmethod", "classmethod", "property"})
# Files that configure how pytest runs every test.
PYTEST_CONFIG_FILES = frozenset({"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py"})


@dataclass
class Selection:
    """Static selection result plus in-memory patched files and hot-patch eligibility."""

    mask: int
    reason: str
    patched_files: dict[str, str]
    # None -> requires Phase 1 clean import; list -> Phase 2 function __code__ hot-patch targets.
    hot_patch_targets: list[tuple[str, str]] | None
    # Phase 2 safe non-import-time literal class-attribute updates: (rel_path, cls_qual, attr, val).
    attr_hot_targets: list[tuple[str, str, str, object]] | None = None
    fixture_only: bool = False
    # Bitmask of tests in `mask` pulled in solely by pushdown_predicates.py:36-37 lazy imports.
    lazy_pp_mask: int = 0
    # Modules whose import ran an edited statement: `mask` holds only if their import-time
    # state digests are unchanged under the patch, else `coarse_mask` applies.
    import_checks: list[str] = field(default_factory=list)
    coarse_mask: int = 0
    # Every change resolved to these sqlglot statement spans (path, first, last), or None when
    # some change was resolved otherwise: lets the verifier rerun only fixture cases that reach them.
    edited_spans: list[tuple[str, int, int]] | None = None
    # Source spans (head_lineno, end_lineno, has_future_annotations) for single-function hot-patch compile.
    hot_patch_spans: dict[tuple[str, str], tuple[int, int, bool]] | None = None
    # Bitmask of tests pulled in by files outside sqlglot/, so case filtering never narrows them.
    non_sqlglot_mask: int = 0
    # Relative fixture path (e.g. "optimizer/simplify.sql") -> changed 0-based pair indices (<=8 pairs).
    fixture_changed_pairs: dict[str, set[int]] | None = None
    # Phase 2 safe leaf-class tracked-dict entry updates:
    # (rel_path, cls_qual, attr, removed_key_srcs, set_item_srcs)
    dict_hot_targets: list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None = None
    # Bitmask of tests pulled in by in-place sqlglot/*.py edits.
    sqlglot_mask: int = 0
    # Bitmask of tests pulled in by optimizer fixtures with extracted changed pairs.
    fixture_mapped_mask: int = 0


def _decorator_name(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _is_safe_default_expr(node: ast.expr | None) -> bool:
    """True if a default-argument expression is a pure literal/container that cannot fail or mutate at import."""
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        return _is_safe_default_expr(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_default_expr(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_safe_default_expr(k) and _is_safe_default_expr(v) for k, v in zip(node.keys, node.values))
    return False


def _has_safe_defaults(func_node: ast.AST) -> bool:
    args = getattr(func_node, "args", None)
    if args is None:
        return False
    return all(_is_safe_default_expr(d) for d in args.defaults) and all(
        _is_safe_default_expr(d) for d in args.kw_defaults if d is not None
    )


def _has_no_annotations(func_node: ast.AST) -> bool:
    if getattr(func_node, "returns", None) is not None:
        return False
    args = getattr(func_node, "args", None)
    if args is None:
        return True
    all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    return all(a.annotation is None for a in all_args)


def _is_overload(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "overload") or (
        isinstance(node, ast.Attribute) and node.attr == "overload"
    )


def _runtime_def(scopes: Scopes, qualname: str):
    """The def a qualname is bound to at runtime: the only one, or the last after @overload stubs."""
    defs = [f for f in scopes.functions if f[3] == qualname]
    if not defs or not all(any(_is_overload(d) for d in f[4].decorator_list) for f in defs[:-1]):
        return None
    return defs[-1]


def _matching_compound_subblocks(
    so: ast.stmt,
    sn: ast.stmt,
    ignore_ann: bool,
) -> list[tuple[list[ast.stmt], list[ast.stmt]]] | None:
    """If two same-type compound statements share an identical header, return their paired sub-blocks."""
    if isinstance(so, (ast.If, ast.While)):
        if canon_ast(so.test, ignore_ann) == canon_ast(sn.test, ignore_ann):
            return [(so.body, sn.body), (so.orelse, sn.orelse)]
    elif isinstance(so, (ast.For, ast.AsyncFor)):
        if canon_ast(so.target, ignore_ann) == canon_ast(sn.target, ignore_ann) and canon_ast(
            so.iter, ignore_ann
        ) == canon_ast(sn.iter, ignore_ann):
            return [(so.body, sn.body), (so.orelse, sn.orelse)]
    elif isinstance(so, (ast.With, ast.AsyncWith)):
        if canon_ast(so.items, ignore_ann) == canon_ast(sn.items, ignore_ann):
            return [(so.body, sn.body)]
    elif isinstance(so, ast.Try):
        if len(so.handlers) == len(sn.handlers) and all(
            canon_ast(ho.type, ignore_ann) == canon_ast(hn.type, ignore_ann) and ho.name == hn.name
            for ho, hn in zip(so.handlers, sn.handlers)
        ):
            pairs = [(so.body, sn.body), (so.orelse, sn.orelse), (so.finalbody, sn.finalbody)]
            pairs.extend((ho.body, hn.body) for ho, hn in zip(so.handlers, sn.handlers))
            return pairs
    return None


def _same_compound_bodies(so: ast.stmt, sn: ast.stmt, ignore_ann: bool) -> bool:
    """True when all sub-block bodies of two same-type compound statements are canonically identical."""
    for field in ("body", "orelse", "finalbody"):
        bo = getattr(so, field, None)
        bn = getattr(sn, field, None)
        if bo is not None or bn is not None:
            if canon_ast(bo, ignore_ann) != canon_ast(bn, ignore_ann):
                return False
    ho = getattr(so, "handlers", None)
    hn = getattr(sn, "handlers", None)
    if ho is not None or hn is not None:
        if canon_ast(ho, ignore_ann) != canon_ast(hn, ignore_ann):
            return False
    return True


def _extract_attr_hot_targets(
    path: str,
    module: str,
    old_scopes: Scopes,
    new_scopes: Scopes,
    hunks,
    impact: ImpactMap,
) -> list[tuple[str, str, str, object]] | None:
    """Return [(path, cls_qual, attr, literal_val)] if every edit only updates non-import-time literal class attrs."""
    targets: list[tuple[str, str, str, object]] = []
    seen: set[tuple[str, str]] = set()
    for removed, added in change_blocks(hunks):
        if not removed and not added:
            continue
        target_cls = target_attr = None
        for old_no, _ in removed:
            info = old_scopes.class_assign_at(old_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        for new_no, _anchor, _ in added:
            info = new_scopes.class_assign_at(new_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        if target_cls is None or target_attr is None:
            return None
        if impact.read_at_import(f"{module}:{target_cls}", target_attr):
            return None
        # A test module read it while importing: the pre-warmed copy would be stale.
        if f"{module}:{target_cls}|{target_attr}" in impact.collect_attrs:
            return None
        if f"{module}:{target_cls}|{target_attr}" not in impact.tracked_attrs:
            return None
        new_stmt = new_scopes.class_assigns.get((target_cls, target_attr))
        if new_stmt is None or new_stmt.value is None:
            return None
        try:
            lit_val = ast.literal_eval(new_stmt.value)
        except Exception:
            return None
        if (target_cls, target_attr) not in seen:
            seen.add((target_cls, target_attr))
            targets.append((path, target_cls, target_attr, lit_val))
    return targets if targets else None


_NO_DICT_HOTPATCH_ATTRS = frozenset({
    "KEYWORDS",
    "SINGLE_TOKENS",
    "TIME_MAPPING",
    "FORMAT_MAPPING",
    "INVERSE_TIME_MAPPING",
    "INVERSE_FORMAT_MAPPING",
    "CREATABLE_KIND_MAPPING",
    "UNESCAPED_SEQUENCES",
    "ESCAPED_SEQUENCES",
    "IDENTIFIER_ESCAPED_SEQUENCES",
    "DATE_PART_MAPPING",
    "JODA_TIME_MAPPING",
    "INVERSE_JODA_TIME_MAPPING",
})


def _bound_names_in(node: ast.AST) -> set[str]:
    """Return local names bound by lambda args or comprehension targets."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Lambda):
            args = sub.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                names.add(a.arg)
            if args.vararg:
                names.add(args.vararg.arg)
            if args.kwarg:
                names.add(args.kwarg.arg)
        elif isinstance(sub, ast.comprehension):
            for tgt in ast.walk(sub.target):
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
    return names


def _safe_eval_expr(node: ast.expr, cls_locals: set[str]) -> bool:
    """True when `node` can be compiled and evaluated directly in the enclosing module's globals."""
    bound = _bound_names_in(node)
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Yield, ast.YieldFrom, ast.Await)):
            return False
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id not in bound and (
                sub.id in ("super", "__class__", "locals", "vars") or sub.id in cls_locals
            ):
                return False
    try:
        compile(ast.unparse(node), "<dict>", "eval")
    except Exception:
        return False
    return True


def _explicit_dict_kv(
    dict_node: ast.Dict,
) -> tuple[dict[tuple[str, ...], tuple[ast.expr, ast.expr]], list[str]] | None:
    """Extract explicit `toks -> (key_node, val_node)` entries plus spread dumps from an ast.Dict."""
    kv: dict[tuple[str, ...], tuple[ast.expr, ast.expr]] = {}
    spreads: list[str] = []
    for k, v in zip(dict_node.keys, dict_node.values):
        if k is None:
            spreads.append(ast.dump(v))
            continue
        toks = _ast_dict_key_tokens(k)
        if toks is None or toks in kv:
            return None
        kv[toks] = (k, v)
    return kv, spreads


def _extract_dict_hot_targets(
    path: str,
    module: str,
    old_scopes: Scopes,
    new_scopes: Scopes,
    hunks,
    impact: ImpactMap,
) -> list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None:
    """Return Phase-2 entry updates if every edit modifies explicit keys of tracked leaf-class dicts."""
    if not impact.leaf_classes:
        return None
    seen_attrs: list[tuple[str, str, ast.stmt, ast.stmt]] = []
    seen_keys: set[tuple[str, str]] = set()
    for removed, added in change_blocks(hunks):
        if not removed and not added:
            continue
        target_cls = target_attr = None
        for old_no, _ in removed:
            info = old_scopes.class_assign_at(old_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        for new_no, _anchor, _ in added:
            info = new_scopes.class_assign_at(new_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        if target_cls is None or target_attr is None:
            return None
        pair = (target_cls, target_attr)
        if pair not in seen_keys:
            old_stmt = old_scopes.class_assigns.get(pair)
            new_stmt = new_scopes.class_assigns.get(pair)
            if old_stmt is None or new_stmt is None:
                return None
            seen_keys.add(pair)
            seen_attrs.append((target_cls, target_attr, old_stmt, new_stmt))

    if not seen_attrs:
        return None

    targets: list[tuple[str, str, str, list[str], list[tuple[str, str]]]] = []
    for cls_qual, attr, old_stmt, new_stmt in seen_attrs:
        if cls_qual == "Tokenizer" or cls_qual.endswith(".Tokenizer") or attr in _NO_DICT_HOTPATCH_ATTRS:
            return None
        cls_key = f"{module}:{cls_qual}"
        key_id = f"{cls_key}|{attr}"
        if (
            key_id not in impact.tracked_dicts
            or cls_key not in impact.leaf_classes
            or cls_key in impact.pre_hook_classes
            or key_id in impact.collect_attrs
        ):
            return None
        old_val = getattr(old_stmt, "value", None)
        new_val = getattr(new_stmt, "value", None)
        if not (isinstance(old_val, ast.Dict) and isinstance(new_val, ast.Dict)):
            return None
        old_parsed = _explicit_dict_kv(old_val)
        new_parsed = _explicit_dict_kv(new_val)
        if old_parsed is None or new_parsed is None:
            return None
        old_kv, old_spreads = old_parsed
        new_kv, new_spreads = new_parsed
        if old_spreads != new_spreads:
            return None
        if [k for k in old_kv if k in new_kv] != [k for k in new_kv if k in old_kv]:
            return None
        removed_toks = [k for k in old_kv if k not in new_kv]
        added_toks = [k for k in new_kv if k not in old_kv]
        modified_toks = [
            k for k in old_kv if k in new_kv and ast.dump(old_kv[k][1]) != ast.dump(new_kv[k][1])
        ]
        if not (removed_toks or added_toks or modified_toks):
            return None
        touched_tokens = [tok for toks in (removed_toks + added_toks + modified_toks) for tok in toks]
        if impact.read_at_import(cls_key, attr, touched_tokens):
            return None

        cls_prefix = cls_qual + "."
        cls_locals = {a for (cq, a) in new_scopes.class_assigns if cq == cls_qual} | {
            f[3][len(cls_prefix) :]
            for f in new_scopes.functions
            if f[3].startswith(cls_prefix) and "." not in f[3][len(cls_prefix) :]
        }
        removed_srcs: list[str] = []
        for toks in removed_toks:
            k_node = old_kv[toks][0]
            if not _safe_eval_expr(k_node, cls_locals):
                return None
            removed_srcs.append(ast.unparse(k_node))
        set_srcs: list[tuple[str, str]] = []
        for toks in added_toks + modified_toks:
            k_node, v_node = new_kv[toks]
            if not (_safe_eval_expr(k_node, cls_locals) and _safe_eval_expr(v_node, cls_locals)):
                return None
            set_srcs.append((ast.unparse(k_node), ast.unparse(v_node)))
        targets.append((path, cls_qual, attr, removed_srcs, set_srcs))
    return targets if targets else None


def _dict_scope_mask(cls_key: str, attr: str, impact: ImpactMap) -> int | None:
    """All tests that use `cls_key`, reference it, or look up any key of `cls_key|attr`."""
    mod, cls_qual = cls_key.split(":", 1)
    cls_usage = impact.class_usage(mod, cls_qual)
    if cls_usage is None:
        return None
    return cls_usage | impact.class_ref_usage(mod, cls_qual) | impact.dict_any_usage(cls_key, attr)


def _narrow_added_dict_key(cls_key: str, attr: str, toks: tuple[str, ...], impact: ImpactMap) -> int | None:
    """Soundly narrow a newly added key in a tracked class dict, or return None."""
    scope = _dict_scope_mask(cls_key, attr, impact)
    if scope is None:
        return None
    exp_name = next((tok[2:] for tok in toks if tok.startswith("t:")), None)
    if exp_name is None:
        return scope
    if impact.pinned_identifiers and exp_name not in impact.pinned_identifiers:
        return 0
    exp_mask = impact.exp_class_usage(exp_name)
    if exp_mask is None:
        return scope
    return scope & exp_mask


_LINE_SQL_FIXTURES = frozenset(
    {
        "tests/fixtures/identity.sql",
        "tests/fixtures/partial.sql",
        "tests/fixtures/optimizer/qualify_columns__invalid.sql",
    }
)


def _filter_sql_comments(text: str) -> str:
    """Exact match to tests/helpers.py::_filter_comments."""
    return "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))


def _fixture_pairs(text: str) -> list[tuple[dict[str, str], str, str]] | None:
    """Parse sqlglot optimizer fixture exactly like tests/helpers.py::load_sql_fixture_pairs.

    Returns None if malformed (e.g. a `#` header line missing `:`), so callers fall back
    conservatively when TestExecutor.setUpClass() would raise.
    """
    try:
        filtered = _filter_sql_comments(text)
        stmts = filtered.split(";")
        out: list[tuple[dict[str, str], str, str]] = []
        for i in range(0, len(stmts) - 1, 2):
            sql = stmts[i].strip()
            sql_lines = sql.split("\n")
            meta: dict[str, str] = {}
            idx = 0
            while sql_lines[idx].startswith("#"):
                key, val = sql_lines[idx].split(":", maxsplit=1)
                meta[key.lstrip("#").strip()] = val.strip()
                idx += 1
            out.append((meta, "\n".join(sql_lines[idx:]), stmts[i + 1].strip()))
        return out
    except Exception:
        return None


def _tpc_fixture_inputs(pairs: list[tuple[dict[str, str], str, str]]) -> list[tuple[dict[str, str], str]]:
    return [(m, s) for m, s, _ in pairs]


def _fixture_pair_key(pair: tuple[dict[str, str], str, str]) -> tuple:
    return (tuple(sorted(pair[0].items())), pair[1], pair[2])


def _extract_hot_patch_targets(
    path: str,
    old_scopes: Scopes,
    new_scopes: Scopes,
    hunks,
    runs_at_import,
    ignore_ann: bool = False,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], tuple[int, int, bool]]] | None:
    """Return ([(path, qualname)], spans) if every edit is strictly inside existing non-nested function bodies."""
    targets: set[tuple[str, str]] = set()
    spans: dict[tuple[str, str], tuple[int, int, bool]] = {}
    fut_ann = new_scopes.has_future_annotations
    for removed, added in change_blocks(hunks):
        for old_no, _ in removed:
            fn = old_scopes.function_at(old_no)
            if fn is None or old_no < fn[0]:
                return None
            body_start, end, _head, qual, old_node = fn
            if "<locals>" in qual or runs_at_import(path, body_start, end):
                return None
            old_rt, new_fn = _runtime_def(old_scopes, qual), _runtime_def(new_scopes, qual)
            if old_rt is None or old_rt[4] is not old_node or new_fn is None:
                return None
            new_node = new_fn[4]
            if not all(_decorator_name(d) in SAFE_HOTPATCH_DECORATORS for d in old_node.decorator_list):
                return None
            if [ast.dump(d) for d in old_node.decorator_list] != [ast.dump(d) for d in new_node.decorator_list]:
                return None
            if canon_ast(old_node.args, ignore_ann) != canon_ast(new_node.args, ignore_ann):
                return None
            targets.add((path, qual))
            spans[(path, qual)] = (new_fn[2], new_fn[1], fut_ann)

        for new_no, _anchor, _ in added:
            fn = new_scopes.function_at(new_no)
            if fn is None or new_no < fn[0]:
                return None
            _body_start, _end, _head, qual, new_node = fn
            if "<locals>" in qual:
                return None
            old_fn, new_rt = _runtime_def(old_scopes, qual), _runtime_def(new_scopes, qual)
            if new_rt is None or new_rt[4] is not new_node:
                return None
            if old_fn is None or runs_at_import(path, old_fn[0], old_fn[1]):
                return None
            old_node = old_fn[4]
            if not all(_decorator_name(d) in SAFE_HOTPATCH_DECORATORS for d in old_node.decorator_list):
                return None
            if [ast.dump(d) for d in old_node.decorator_list] != [ast.dump(d) for d in new_node.decorator_list]:
                return None
            if canon_ast(old_node.args, ignore_ann) != canon_ast(new_node.args, ignore_ann):
                return None
            targets.add((path, qual))
            spans[(path, qual)] = (new_rt[2], new_rt[1], fut_ann)
    return sorted(targets), spans


class Selector:
    def __init__(self, impact: ImpactMap):
        self.impact = impact
        self._old: dict[str, tuple[str, Scopes]] = {}
        self._no_pp_lazy: bool = False
        self._hit_pp_lazy: bool = False
        self._coarse_block: bool = False
        self._no_import_check: bool = False
        self._no_verify: bool = False
        self._spans: list[tuple[str, int, int]] = []
        self._spans_ok: bool = True
        self._block_used_values: bool = False
        self._owner_checks: set[str] = set()
        self._fixture_changed_pairs: dict[str, set[int]] = {}
        self._state = marshal.loads(STATE_FILE.read_bytes()) if STATE_FILE.is_file() else None
        self.values = None
        values_file = APP / "map" / "values.bin"
        if values_file.is_file() and os.environ.get("BR_VALUES", "1") != "0":
            from blast_radius.values import ValueMap

            self.values = ValueMap(values_file, impact.test_ids)
        self._registry_bits = sum(
            1 << i for i, t in enumerate(impact.test_ids) if t == "tests/dialects/test_dialect.py::TestDialect::test_enum"
        )
        docs = [i for i, t in enumerate(impact.test_ids) if t.startswith("tests/test_docs.py")]
        self.docs_mask = sum(1 << i for i in docs)
        exec_cls_mask = impact.by_class.get("tests/test_executor.py::TestExecutor", 0)
        tpcds_exec_bit = sum(
            1 << i
            for i, t in enumerate(impact.test_ids)
            if t == "tests/test_executor.py::TestExecutor::test_execute_tpcds"
        )
        tpch_exec_bit = sum(
            1 << i
            for i, t in enumerate(impact.test_ids)
            if t == "tests/test_executor.py::TestExecutor::test_execute_tpch"
        )
        tpch_opt_bit = sum(
            1 << i
            for i, t in enumerate(impact.test_ids)
            if t == "tests/test_executor.py::TestExecutor::test_optimized_tpch"
        )
        # TestExecutor.setUpClass loads both tpc-ds.sql and tpc-h.sql, attributing both files to
        # all 60 TestExecutor tests. When the patched fixture still parses cleanly in setUpClass,
        # only the tests that actually read that suite's sqls can be affected.
        self._tpc_fixture_strip: dict[str, tuple[int, int]] = {
            # (always_strip_when_parseable, extra_strip_when_inputs_unchanged)
            "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": (
                exec_cls_mask & ~tpcds_exec_bit,
                tpcds_exec_bit,
            ),
            "tests/fixtures/optimizer/tpc-h/tpc-h.sql": (
                exec_cls_mask & ~(tpch_exec_bit | tpch_opt_bit),
                tpch_exec_bit,
            ),
        }

    def old_source(self, path: str) -> tuple[str, Scopes]:
        if path not in self._old:
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned source for {path}")
            text = file.read_text(encoding="utf-8")
            self._old[path] = (text, Scopes(ast.parse(text), has_lambda="lambda" in text))
        return self._old[path]

    def select(self, patch_text: str) -> tuple[int, str]:
        sel = self.select_detailed(patch_text)
        return sel.mask, sel.reason

    def select_detailed(self, patch_text: str) -> Selection:
        impact = self.impact
        try:
            files = parse_patch(patch_text)
            if not files:
                raise Everything("empty patch")
            mask = 0
            sqlglot_mask = 0
            fixture_mapped_mask = 0
            non_sqlglot_mask = 0
            lazy_pp_mask = 0
            all_noop = True
            all_fixture_or_noop = True
            patched_files: dict[str, str] = {}
            hot_targets: list[tuple[str, str]] | None = []
            hot_spans: dict[tuple[str, str], tuple[int, int, bool]] = {}
            attr_targets: list[tuple[str, str, str, object]] | None = []
            dict_targets: list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None = []
            self._hit_pp_lazy = False
            self._owner_checks = set()
            self._no_verify = False
            self._spans, self._spans_ok = [], True
            self._fixture_changed_pairs = {}
            self._hot_spans_cur: dict[tuple[str, str], tuple[int, int, bool]] = {}
            for file_patch in files:
                self._hot_spans_cur = {}
                file_mask, is_noop, is_fixture_or_comment, f_targets, a_targets, d_targets = self.select_file_detailed(
                    file_patch, patched_files
                )
                mask |= file_mask
                if file_patch.new and file_patch.old == file_patch.new and file_patch.new.startswith("sqlglot/"):
                    sqlglot_mask |= file_mask
                elif (
                    file_patch.new
                    and file_patch.old == file_patch.new
                    and file_patch.new.startswith("tests/fixtures/")
                    and file_patch.new[len("tests/fixtures/") :] in self._fixture_changed_pairs
                ):
                    fixture_mapped_mask |= file_mask
                else:
                    non_sqlglot_mask |= file_mask
                all_noop = all_noop and is_noop
                all_fixture_or_noop = all_fixture_or_noop and is_fixture_or_comment
                if f_targets is None and a_targets is None and d_targets is None:
                    hot_targets = None
                    attr_targets = None
                    dict_targets = None
                else:
                    if hot_targets is not None and f_targets:
                        hot_targets.extend(f_targets)
                        hot_spans.update(self._hot_spans_cur)
                    if attr_targets is not None and a_targets:
                        attr_targets.extend(a_targets)
                    if dict_targets is not None and d_targets:
                        dict_targets.extend(d_targets)
            # The extra passes below re-select with other settings: keep this pass's spans and pairs.
            spans = (
                [s for s in dict.fromkeys(self._spans) if s[0].startswith("sqlglot/")]
                if self._spans_ok
                else None
            )
            changed_pairs = dict(self._fixture_changed_pairs) if self._fixture_changed_pairs else None
            if self._hit_pp_lazy and mask:
                self._no_pp_lazy = True
                try:
                    mask_no_pp = 0
                    for file_patch in files:
                        fm, _, _, _, _, _ = self.select_file_detailed(file_patch, {})
                        mask_no_pp |= fm
                    lazy_pp_mask = mask & ~mask_no_pp
                finally:
                    self._no_pp_lazy = False
            owners = sorted(self._owner_checks)
            coarse_mask = 0
            if owners:
                self._no_import_check = True
                try:
                    for file_patch in files:
                        fm, _, _, _, _, _ = self.select_file_detailed(file_patch, {})
                        coarse_mask |= fm
                finally:
                    self._no_import_check = False
                if self._no_verify:
                    # The import check runs the patched code, which this commit cannot overlay.
                    mask, owners, lazy_pp_mask = coarse_mask, [], 0
            if owners and self._registry_bits:
                # The edited code runs while modules import, i.e. on the paths that lazily load
                # dialects. test_enum reads the global dialect registry that earlier tests filled:
                # its outcome depends on other tests' calls, which per-test coverage cannot show.
                mask |= self._registry_bits
                coarse_mask |= self._registry_bits
            reason = "noop" if all_noop and mask == 0 else "mapped"
            return Selection(
                mask=mask,
                reason=reason,
                # No patched files means no runtime verification (deleted/renamed/binary files).
                patched_files={} if self._no_verify else patched_files,
                hot_patch_targets=hot_targets,
                attr_hot_targets=attr_targets,
                fixture_only=all_fixture_or_noop and not all_noop,
                lazy_pp_mask=lazy_pp_mask,
                import_checks=owners,
                coarse_mask=coarse_mask,
                edited_spans=spans,
                hot_patch_spans=hot_spans if hot_targets else None,
                non_sqlglot_mask=non_sqlglot_mask,
                fixture_changed_pairs=changed_pairs,
                dict_hot_targets=dict_targets,
                sqlglot_mask=sqlglot_mask,
                fixture_mapped_mask=fixture_mapped_mask,
            )
        except Everything as reason:
            return Selection(impact.everything, f"everything ({reason})", {}, None, None, False)
        except Exception as error:  # never let an unexpected input crash the run
            return Selection(
                impact.everything,
                f"everything (error {type(error).__name__}: {error})",
                {},
                None,
                None,
                False,
            )

    def select_file_detailed(
        self,
        fp: FilePatch,
        patched_files: dict[str, str],
    ) -> tuple[
        int,
        bool,
        bool,
        list[tuple[str, str]] | None,
        list[tuple[str, str, str, object]] | None,
        list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None,
    ]:
        impact = self.impact
        is_sqlglot = bool(
            (fp.new and fp.new.startswith("sqlglot/")) or (fp.old and fp.old.startswith("sqlglot/"))
        )
        if is_sqlglot and not (fp.new and fp.new == fp.old and not fp.binary and fp.new.endswith(".py")):
            self._spans_ok = False  # only in-place sqlglot .py edits resolve to statement spans
        if fp.binary:
            self._no_verify = True
            mask = 0
            for p in {fp.old, fp.new} - {None}:
                mask |= self._readers(p)
                if p.endswith(".py") and p.startswith(("sqlglot/", "tests/")) and p == fp.old:
                    mask |= impact.module_level(p)
            return mask, False, False, None, None, None
        if (fp.old is not None and fp.new is not None and fp.old != fp.new) or fp.new is None:
            # Deleted or renamed non-binary file: represent deletion as None in patched_files
            # so Phase-1 virtual overlay raises ModuleNotFoundError/FileNotFoundError.
            mask = 0
            if fp.old is not None:
                patched_files[fp.old] = None  # type: ignore[assignment]
                mask |= self._readers(fp.old)
                if fp.old.endswith(".py") and fp.old.startswith(("sqlglot/", "tests/")):
                    mask |= impact.module_level(fp.old)
            if fp.new is not None:
                old_file = SOURCE / fp.old if fp.old else None
                base_text = old_file.read_text(encoding="utf-8") if (old_file and old_file.is_file()) else ""
                patched_files[fp.new] = apply(base_text, fp.hunks)
                mask |= self._readers(fp.new)
            return mask, False, False, None, None, None
        path = fp.new
        if fp.old is None:
            # New file: no pinned code imports or opens it; tests that list its folder may see it.
            # The overlay serves it, so verification still works.
            patched_files[path] = apply("", fp.hunks)
            return self._readers(path), False, False, None, None, None
        if path.startswith("tests/fixtures/"):
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned fixture for {path}")
            old_text = file.read_text(encoding="utf-8")
            new_text = apply(old_text, fp.hunks)
            old_pairs = new_pairs = None
            if path in _LINE_SQL_FIXTURES:
                # load_sql_fixtures strips `--` comments and blank lines.
                if _filter_sql_comments(old_text) == _filter_sql_comments(new_text):
                    return 0, True, True, [], [], []
            elif (path.startswith("tests/fixtures/optimizer/") or path == "tests/fixtures/pretty.sql") and path.endswith(
                ".sql"
            ):
                old_pairs = _fixture_pairs(old_text)
                new_pairs = _fixture_pairs(new_text)
                # load_sql_fixture_pairs strips `--` comments & blank lines and trims around `;`:
                # pure comment/whitespace fixture edits cannot change any test.
                if old_pairs is not None and new_pairs is not None and old_pairs == new_pairs:
                    return 0, True, True, [], [], []
            patched_files[path] = new_text
            if (
                path.startswith("tests/fixtures/optimizer/")
                and path not in _LINE_SQL_FIXTURES
                and path.endswith(".sql")
                and old_pairs is not None
                and new_pairs is not None
                and not self._no_pp_lazy
                and not self._no_import_check
            ):
                if len(old_pairs) == len(new_pairs):
                    changed_idx = {idx for idx, (o, n) in enumerate(zip(old_pairs, new_pairs)) if o != n}
                else:
                    old_key_set = {_fixture_pair_key(p) for p in old_pairs}
                    changed_idx = {idx for idx, p in enumerate(new_pairs) if _fixture_pair_key(p) not in old_key_set}
                is_tpc_fix = path in (
                    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql",
                    "tests/fixtures/optimizer/tpc-h/tpc-h.sql",
                )
                if 1 <= len(changed_idx) <= 8 or (
                    len(changed_idx) == 0 and len(new_pairs) > 0 and not is_tpc_fix
                ):
                    self._fixture_changed_pairs[path[len("tests/fixtures/") :]] = changed_idx
            fmask = self.select_fixture(path)
            strip_spec = self._tpc_fixture_strip.get(path)
            if strip_spec is not None and old_pairs is not None and new_pairs is not None:
                # setUpClass parses new_text cleanly: strip unrelated TestExecutor tests (e.g.
                # editing tpc-h.sql can never break test_execute_tpcds!).
                always_strip, inputs_unchanged_strip = strip_spec
                fmask &= ~always_strip
                if _tpc_fixture_inputs(old_pairs) == _tpc_fixture_inputs(new_pairs):
                    fmask &= ~inputs_unchanged_strip
            return fmask, False, True, [], [], []
        if path == "pyproject.toml":
            file = SOURCE / path
            if file.is_file():
                old_toml_text = file.read_text(encoding="utf-8")
                new_toml_text = apply(old_toml_text, fp.hunks)
                try:
                    old_toml = tomllib.loads(old_toml_text)
                    new_toml = tomllib.loads(new_toml_text)
                    old_tool = old_toml.get("tool") if isinstance(old_toml.get("tool"), dict) else {}
                    new_tool = new_toml.get("tool") if isinstance(new_toml.get("tool"), dict) else {}
                    pytest_same = old_tool.get("pytest") == new_tool.get("pytest") and "pytest" not in old_toml and "pytest" not in new_toml
                except Exception:
                    pytest_same = False
                if pytest_same:
                    readers = self._readers(path)
                    if readers:
                        patched_files[path] = new_toml_text
                    return readers, readers == 0, readers == 0, [], [], []
        if path in PYTEST_CONFIG_FILES:
            raise Everything(f"pytest configuration {path}")
        if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
            if path.startswith("tests/") and (path.endswith("__init__.py") or path.endswith("conftest.py")):
                # Package plumbing: a real change reaches every test collected under its folder.
                saved_spans_ok = self._spans_ok
                py_mask, py_noop, new_text, _, _, _ = self.select_python_detailed(path, fp.hunks)
                self._spans_ok = saved_spans_ok
                if py_noop:
                    return 0, True, True, [], [], []
                patched_files[path] = new_text
                folder = path.rsplit("/", 1)[0] + "/"
                under = sum(1 << i for i, t in enumerate(impact.test_ids) if t.startswith(folder))
                return under | py_mask, False, False, None, None, None
            if path.startswith("sqlglot/") and path not in impact.lines and path not in impact.importers:
                # Never imported or run by any test: only tests that open the file can notice.
                return self._readers(path), self._readers(path) == 0, False, [], [], []
            # Always include tests that directly opened this .py file (e.g. test_typing.py),
            # except when a sqlglot/typing/*.py change is a pure comment or AST-identical noop
            # (test_no_redundant_expression_metadata only inspects AST imports and EXPRESSION_METADATA).
            readers = impact.file_readers(path)
            saved_spans_ok = self._spans_ok
            py_mask, py_noop, new_text, targets, attr_targets, dict_targets = self.select_python_detailed(
                path, fp.hunks
            )
            if not path.startswith("sqlglot/"):
                self._spans_ok = saved_spans_ok
            if py_noop and path.startswith("sqlglot/typing/"):
                readers = 0
            if not py_noop or readers:
                patched_files[path] = new_text
            return readers | py_mask, py_noop and readers == 0, py_noop, targets, attr_targets, dict_targets
        # Docs, packaging, CI config, the Rust tokenizer (not installed) ...: only tests that
        # open the file or list its folder can notice a change.
        readers = self._readers(path)
        if readers:
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned source for {path}")
            patched_files[path] = apply(file.read_text(encoding="utf-8"), fp.hunks)
        return readers, readers == 0, readers == 0, [], [], []

    def _readers(self, path: str) -> int:
        """Tests that opened the file or listed its folder in the pinned run."""
        impact = self.impact
        mask = 0
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        for key in ("f:" + path, "d:" + folder):
            index = impact.paths.get(key)
            if index is not None:
                mask |= impact.mask(index)
        return mask

    def select_fixture(self, path: str) -> int:
        # A fixture no test opened (nor listed its folder) cannot change any test's outcome.
        return self._readers(path)

    def select_python_detailed(
        self,
        path: str,
        hunks,
    ) -> tuple[
        int,
        bool,
        str,
        list[tuple[str, str]] | None,
        list[tuple[str, str, str, object]] | None,
        list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None,
    ]:
        old_text, old_scopes = self.old_source(path)
        new_text = apply(old_text, hunks)

        changed = [line[1:] for _, _, _, body in hunks for line in body if line[0] in "+-"]
        if comment_lines_only(changed) and same_code(old_text, new_text):
            return 0, True, new_text, [], [], []
        try:
            new_tree = ast.parse(new_text)
            compile(new_tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            # The module cannot import: exactly what an import-time change reaches fails
            # (plus test_docs.py for sqlglot/ modules walked dynamically by pkgutil).
            self._spans_ok = False
            mask = self.impact.module_level(path)
            if path.startswith("sqlglot/"):
                mask |= self.docs_mask
            return mask, False, new_text, None, None, None
        new_scopes = Scopes(new_tree, has_lambda="lambda" in new_text)

        # Normalized AST equivalence catches quote/paren/wrapping refactors, TYPE_CHECKING-only
        # edits, and pure type-annotation edits when `from __future__ import annotations` holds.
        blocks = list(change_blocks(hunks))
        changed_old_lines = {ln for rem, _ in blocks for ln, _ in rem}
        changed_new_lines = {ln for _, add in blocks for ln, _, _ in add}
        ignore_ann = (
            old_scopes.has_future_annotations
            and new_scopes.has_future_annotations
            and self.impact.file_readers(path) == 0
        )
        if same_ast(
            old_scopes.tree,
            new_tree,
            old_text,
            new_text,
            ignore_annotations=ignore_ann,
            old_lines=changed_old_lines,
            new_lines=changed_new_lines,
        ):
            return 0, True, new_text, [], [], []

        module = module_name(path)
        mask = 0
        self._coarse_block = False
        all_blocks_used_values = bool(blocks)
        owners_before = set(self._owner_checks)
        for removed, added in blocks:
            self._block_used_values = False
            mask |= self.select_block(path, module, old_scopes, new_scopes, removed, added, ignore_ann)
            all_blocks_used_values = all_blocks_used_values and self._block_used_values

        if (
            self.values is not None
            and mask
            and path.startswith("sqlglot/")
            and (not all_blocks_used_values or self._owner_checks != owners_before)
        ):
            # Every difference is a rearranged condition over the same operands: replay the
            # pinned run's recorded outcomes (import-time ones included) through the edit.
            try:
                vmask = self.values.narrow(path, old_scopes.tree, new_tree, self.impact, hunks)
            except Exception as error:  # noqa: BLE001 - fall back to the rules above
                log(f"value map error on {path}: {type(error).__name__}: {error}")
                vmask = None
            if vmask is not None:
                # Both are sound supersets of what the edit reaches: keep their intersection.
                log(f"{path}: value map narrows {bin(mask).count('1')} -> {bin(mask & vmask).count('1')} tests")
                mask &= vmask
                self._owner_checks = owners_before
                self._coarse_block = False

        targets: list[tuple[str, str]] | None = None
        attr_targets: list[tuple[str, str, str, object]] | None = None
        dict_targets: list[tuple[str, str, str, list[str], list[tuple[str, str]]]] | None = None
        if path.startswith("sqlglot/"):
            # Every block resolved per statement means no edited statement ran before the tests, or
            # only in imports whose built state the runtime proves unchanged: either way, swapping
            # __code__ after a pre-warmed import matches importing the patched module.
            ran_before = self.runs_before_tests if self._coarse_block else (lambda *_: False)
            hp_res = _extract_hot_patch_targets(path, old_scopes, new_scopes, hunks, ran_before, ignore_ann)
            if hp_res is not None:
                targets, hspans = hp_res
                self._hot_spans_cur.update(hspans)
            else:
                attr_targets = _extract_attr_hot_targets(path, module, old_scopes, new_scopes, hunks, self.impact)
                if attr_targets is None:
                    dict_targets = _extract_dict_hot_targets(
                        path, module, old_scopes, new_scopes, hunks, self.impact
                    )
        else:
            # Non-sqlglot .py edits (e.g. tests/*.py) are forced to Phase 1 in verify.py; keeping
            # empty lists here preserves sqlglot/ hot_patch_targets for fast_a and tokenizer checks.
            targets, attr_targets, dict_targets = [], [], []
        return mask, False, new_text, targets, attr_targets, dict_targets

    def select_block(
        self,
        path,
        module,
        old_scopes: Scopes,
        new_scopes: Scopes,
        removed,
        added,
        ignore_ann: bool = False,
    ) -> int:
        # Edits strictly inside `if TYPE_CHECKING:` bodies never execute at runtime.
        if (removed or added) and (
            all(old_scopes.in_type_checking(n) for n, _ in removed)
            and all(new_scopes.in_type_checking(n) for n, _, _ in added)
        ):
            return 0

        # Try fine-grained module-level EXPRESSION_METADATA dict-key tracking for sqlglot/typing/*.py.
        if path in self.impact.typing_dialects:
            t_mask = self.typing_dict_level(path, old_scopes, new_scopes, removed, added)
            if t_mask is not None:
                self._spans_ok = False
                return t_mask

        # Try fine-grained class dict-key or class attribute tracking first.
        attr_mask = self.class_attr_level(path, module, old_scopes, new_scopes, removed, added, ignore_ann)
        if attr_mask is not None:
            return attr_mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        # One-for-one edit of statements inside one function: the tests that ran them.
        if removed and len(removed) == len(added):
            mask = self.statement_level(path, old_scopes, new_scopes, removed, added)
            if mask is not None:
                return mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        # Multi-line edits or insertions confined to a single function body: AST statement diff.
        ast_mask = self.ast_statement_level(path, old_scopes, new_scopes, removed, added, ignore_ann)
        if ast_mask is not None:
            return ast_mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        self._coarse_block = True
        mask = 0
        for lineno, _ in removed:
            mask |= self.old_line(path, module, old_scopes, new_scopes, lineno)
        for new_lineno, anchor, _ in added:
            mask |= self.new_line(path, module, old_scopes, new_scopes, new_lineno, anchor)
        return mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

    def _test_file_own_mask(self, path: str, old_scopes: Scopes, new_scopes: Scopes, removed, added) -> int:
        """When editing a test file, always include the touched test method(s) or test class."""
        if not path.startswith("tests/"):
            return 0
        impact = self.impact
        mask = 0
        for scopes, lines in ((old_scopes, [n for n, _ in removed]), (new_scopes, [n for n, _, _ in added])):
            for lineno in lines:
                fn = scopes.function_at(lineno)
                if fn is not None:
                    qual = fn[3].split(".<locals>.", 1)[0]
                    bit = impact.by_test_qualname.get((path, qual))
                    if bit:
                        mask |= bit
                        continue
                    # Helper method inside a test class that already has line coverage: keep narrow coverage.
                    if impact.span(path, fn[0], fn[1]):
                        continue
                cls = scopes.class_at(lineno)
                if cls is not None:
                    cls_mask = impact.by_class.get(f"{path}::{cls[2].replace('.', '::')}", 0)
                    mask |= cls_mask
        return mask

    def typing_dict_level(self, path: str, old_scopes: Scopes, new_scopes: Scopes, removed, added) -> int | None:
        """Resolve an edit confined to top-level EXPRESSION_METADATA in sqlglot/typing/*.py by dict key."""
        if not removed and not added:
            return None
        for old_no, _ in removed:
            info = old_scopes.module_assign_at(old_no)
            if info is None or info[0] != "EXPRESSION_METADATA":
                return None
        for new_no, _anchor, _ in added:
            info = new_scopes.module_assign_at(new_no)
            if info is None or info[0] != "EXPRESSION_METADATA":
                return None
        old_stmt = old_scopes.module_assigns.get("EXPRESSION_METADATA")
        new_stmt = new_scopes.module_assigns.get("EXPRESSION_METADATA")
        if old_stmt is None or new_stmt is None:
            return None
        old_parsed = _extract_dict_entries(old_stmt.value)
        new_parsed = _extract_dict_entries(new_stmt.value)
        if old_parsed is None or new_parsed is None:
            return None
        old_entries, old_spreads = old_parsed
        new_entries, new_spreads = new_parsed
        if old_spreads != new_spreads:
            return None
        # _TrackedDict records key hits, not misses: key removals, reorderings, and value
        # edits are exact; newly added `exp.<Cls>` keys narrow via (attr_usage & exp_class_usage).
        added_keys = [toks for toks in new_entries if toks not in old_entries]
        changed_keys = [
            toks
            for toks in old_entries
            if toks not in new_entries or ast.dump(old_entries[toks]) != ast.dump(new_entries[toks])
        ]
        same_order = list(old_entries) == list(new_entries)
        if not changed_keys and not added_keys and same_order:
            return 0
        impact = self.impact
        mask = 0
        changed_tokens = [t for toks in changed_keys for t in toks]
        added_tokens = [t for toks in added_keys for t in toks]
        for cls_key in impact.typing_dialects.get(path, ()):
            if not same_order and (
                cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, "EXPRESSION_METADATA", [])
            ):
                return None
            if changed_tokens and impact.read_at_import(cls_key, "EXPRESSION_METADATA", changed_tokens):
                return None
            if added_keys:
                if cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, "EXPRESSION_METADATA", added_tokens):
                    return None
                for toks in added_keys:
                    narrowed = _narrow_added_dict_key(cls_key, "EXPRESSION_METADATA", toks, impact)
                    if narrowed is None:
                        return None
                    mask |= narrowed
            mask |= impact.dict_key_usage(cls_key, "EXPRESSION_METADATA", "*")
            for toks in changed_keys:
                for tok in toks:
                    mask |= impact.dict_key_usage(cls_key, "EXPRESSION_METADATA", tok)
        return mask

    def class_attr_level(self, path, module, old_scopes: Scopes, new_scopes: Scopes, removed, added, ignore_ann: bool = False) -> int | None:
        """Resolve an edit confined to one direct class-body attribute assignment."""
        impact = self.impact
        if not removed and not added:
            return None

        # Verify every edited line in this block sits inside the exact same (class, attr) assignment.
        target_old = None
        for old_no, _ in removed:
            info = old_scopes.class_assign_at(old_no)
            if info is None:
                return None
            if target_old is None:
                target_old = info
            elif (info[0], info[1]) != (target_old[0], target_old[1]):
                return None

        target_new = None
        for new_no, _anchor, _ in added:
            info = new_scopes.class_assign_at(new_no)
            if info is None:
                return None
            if target_new is None:
                target_new = info
            elif (info[0], info[1]) != (target_new[0], target_new[1]):
                return None

        if target_old is None and target_new is not None:
            cls_qual, attr, new_stmt = target_new
            old_stmt = old_scopes.class_assigns.get((cls_qual, attr))
            if old_stmt is None:
                return None
        elif target_new is None and target_old is not None:
            cls_qual, attr, old_stmt = target_old
            new_stmt = new_scopes.class_assigns.get((cls_qual, attr))
            if new_stmt is None:
                return None
        elif target_old is not None and target_new is not None:
            if (target_old[0], target_old[1]) != (target_new[0], target_new[1]):
                return None
            cls_qual, attr, old_stmt = target_old
            new_stmt = target_new[2]
        else:
            return None

        # If every edited line actually lives inside a lambda body within the assignment,
        # let the normal lambda/statement coverage handle it (often even narrower).
        if removed and all(old_scopes.lambda_at(n) is not None and old_scopes.lambda_at(n)[0] < n for n, _ in removed):
            if all(new_scopes.lambda_at(n) is not None for n, _, _ in added):
                return None

        cls_key = f"{module}:{cls_qual}"
        key_id = f"{cls_key}|{attr}"

        # Case 1: Dict-literal attribute with identical key set and no import-time value reads.
        old_val = old_stmt.value
        new_val = new_stmt.value
        if isinstance(old_val, ast.Dict) and isinstance(new_val, ast.Dict):
            if key_id in impact.tracked_dicts:
                old_parsed = _extract_dict_entries(old_val)
                new_parsed = _extract_dict_entries(new_val)
                if old_parsed is not None and new_parsed is not None:
                    old_entries, old_spreads = old_parsed
                    new_entries, new_spreads = new_parsed
                    # _TrackedDict records key hits, not misses: key removals, reorderings (when
                    # no import-time key iteration exists), and value edits are exact; newly added
                    # `exp.<Cls>` keys narrow via (attr_usage & exp_class_usage).
                    if old_spreads == new_spreads:
                        added_keys = [toks for toks in new_entries if toks not in old_entries]
                        same_order = list(old_entries) == list(new_entries)
                        if not same_order and (
                            cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, attr, [])
                        ):
                            return None
                        changed_keys = [
                            toks
                            for toks in old_entries
                            if toks not in new_entries or ast.dump(old_entries[toks]) != ast.dump(new_entries[toks])
                        ]
                        if not changed_keys and not added_keys and same_order:
                            return 0
                        changed_tokens = [tok for toks in changed_keys for tok in toks]
                        if changed_tokens and impact.read_at_import(cls_key, attr, changed_tokens):
                            return None
                        if attr == "TRANSFORMS" and cls_key in impact.pre_hook_classes:
                            return None
                        collect_lines = impact.collect_lines.get(path, ())

                        def _is_lambda_body_edit(toks: tuple[str, ...]) -> bool:
                            old_e = old_entries.get(toks)
                            new_e = new_entries.get(toks)
                            return (
                                isinstance(old_e, ast.Lambda)
                                and isinstance(new_e, ast.Lambda)
                                and canon_ast(old_e.args, ignore_ann) == canon_ast(new_e.args, ignore_ann)
                                and not any(
                                    old_e.body.lineno <= ln <= old_e.body.end_lineno for ln in collect_lines
                                )
                            )

                        mask = impact.dict_key_usage(cls_key, attr, "*")
                        if added_keys:
                            added_tokens = [tok for toks in added_keys for tok in toks]
                            if cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, attr, added_tokens):
                                return None
                            for toks in added_keys:
                                narrowed = _narrow_added_dict_key(cls_key, attr, toks, impact)
                                if narrowed is None:
                                    return None
                                mask |= narrowed
                        for toks in changed_keys:
                            if attr == "TRANSFORMS":
                                # Generator.__init__ caches cls.TRANSFORMS into _DISPATCH_CACHE once per class,
                                # so dict_key_usage only credits the first test; use lambda body lines or scope+exp.
                                if _is_lambda_body_edit(toks):
                                    body = old_entries[toks].body
                                    mask |= impact.span(path, body.lineno, body.end_lineno)
                                else:
                                    narrowed = _narrow_added_dict_key(cls_key, attr, toks, impact)
                                    if narrowed is None:
                                        return None
                                    mask |= narrowed
                            else:
                                for tok in toks:
                                    mask |= impact.dict_key_usage(cls_key, attr, tok)
                        lambda_spans_only = (
                            not added_keys
                            and same_order
                            and bool(changed_keys)
                            and all(_is_lambda_body_edit(toks) for toks in changed_keys)
                        )
                        if lambda_spans_only:
                            for toks in changed_keys:
                                body = old_entries[toks].body
                                self._spans.append((path, body.lineno, body.end_lineno))
                        else:
                            self._spans_ok = False
                        return mask
            return None

        # Case 2: Non-dict class data attribute not read at import/metaclass time.
        if key_id in impact.tracked_attrs and not impact.read_at_import(cls_key, attr):
            self._spans_ok = False
            return impact.attr_usage(cls_key, attr)

        return None

    def statement_level(self, path, old_scopes, new_scopes, removed, added):
        mask = 0
        checked_funcs: set[str] = set()
        old_lines = {n for n, _ in removed}
        new_lines = {n for n, _, _ in added}
        all_narrowed = bool(removed)
        for (old_no, old_text), (new_no, _anchor, new_text) in zip(removed, added):
            if indent(old_text) != indent(new_text):
                return None
            function = old_scopes.function_at(old_no)
            new_function = new_scopes.function_at(new_no)
            if function is None or new_function is None or function[3] != new_function[3]:
                return None
            body_start, end, head, qualname, old_node = function
            if old_no < body_start:
                return None  # def header or decorator
            # Reject statement-level only if non-local lexical bindings or generator flags changed.
            if qualname not in checked_funcs:
                if not same_local_binding_semantics(old_node, new_function[4], old_lines, new_lines):
                    return None
                checked_funcs.add(qualname)
            if old_no in old_scopes.docstrings:
                mask |= self.docs_mask
                all_narrowed = False
                continue
            statement = old_scopes.statement_at(old_no)
            if statement is None:
                return None
            start, stop, stmt = statement
            if isinstance(stmt, COMPOUND_NODES):
                last = header_end(stmt)
                if old_no > last:
                    return None  # else:/finally: lines and the like
                stop = last
            # Per-statement value-map narrowing (also works in multi-hunk files).
            if self.values is not None and path.startswith("sqlglot/"):
                new_statement = new_scopes.statement_at(new_no)
                if new_statement is not None and type(new_statement[2]) is type(stmt):
                    new_stmt = new_statement[2]
                    if not any(start <= ln <= stop for ln in self.impact.collect_lines.get(path, ())):
                        vmask = None
                        try:
                            if isinstance(stmt, (ast.If, ast.While)):
                                vmask = self.values.narrow_expr(path, stmt.test, new_stmt.test, stmt, "test", self.impact)
                            elif not isinstance(stmt, COMPOUND_NODES):
                                vmask = self.values.narrow_stmt(path, stmt, new_stmt, self.impact)
                        except Exception:
                            vmask = None
                        if vmask is not None:
                            self._spans.append((path, start, stop))
                            mask |= vmask & self.impact.span(path, start, stop)
                            continue
            all_narrowed = False
            span = self._test_time_span(path, start, stop)
            if span is None:
                return None
            mask |= span
        if all_narrowed:
            self._block_used_values = True
        return mask

    def _test_time_span(self, path: str, first: int, last: int) -> int | None:
        """Tests that ran an edited statement, or None if it also ran while importing/collecting.

        The map imports every sqlglot module before any test, so a statement that never ran
        then only ever ran inside tests, and is credited to them. Execution only diverges
        once it reaches an edited statement: if that never happened at import or collection
        time, import-time state is unchanged even when the enclosing function ran then.
        """
        if self.runs_before_tests(path, first, last):
            owners = self._import_owners(path, first, last)
            if owners is None:
                return None
            # Ran while other modules imported: tests only see that through the state those
            # imports built, which the runtime re-checks (else the coarse selection is used,
            # and hot-patching is dropped).
            self._owner_checks |= owners
        self._spans.append((path, first, last))
        return self.impact.span(path, first, last)

    def _import_owners(self, path: str, first: int, last: int) -> set[str] | None:
        """Modules whose import ran these lines, if all have a recorded state digest; else None."""
        if self._no_import_check or not self._state:
            return None
        if any(first <= n <= last for n in self.impact.collect_lines.get(path, ())):
            return None
        per_line = self._state["owners"].get(path, {})
        owners: set[str] = set()
        for n in self.impact.import_lines.get(path, ()):
            if first <= n <= last:
                names = per_line.get(n)
                if not names or any(name not in self._state["digests"] for name in names):
                    return None
                owners.update(names)
        return owners

    def ast_statement_level(
        self,
        path: str,
        old_scopes: Scopes,
        new_scopes: Scopes,
        removed,
        added,
        ignore_ann: bool,
    ) -> int | None:
        """Map multi-line edits or insertions inside one function body via recursive AST statement diff."""
        if not removed and not added:
            return None
        qualname = None
        old_fn = None
        new_fn = None
        for old_no, _ in removed:
            fn = old_scopes.function_at(old_no)
            if fn is None or old_no < fn[0]:
                return None
            if qualname is None:
                qualname, old_fn = fn[3], fn
            elif fn[3] != qualname:
                return None
        for new_no, _anchor, _ in added:
            fn = new_scopes.function_at(new_no)
            if fn is None or new_no < fn[0]:
                return None
            if qualname is None:
                qualname, new_fn = fn[3], fn
            elif fn[3] != qualname:
                return None
            else:
                new_fn = fn
        if qualname is None:
            return None
        old_rt, new_rt = _runtime_def(old_scopes, qualname), _runtime_def(new_scopes, qualname)
        if old_rt is None or new_rt is None:
            return None
        if (old_fn is not None and old_fn[4] is not old_rt[4]) or (new_fn is not None and new_fn[4] is not new_rt[4]):
            return None
        old_fn, new_fn = old_rt, new_rt
        old_node = old_fn[4]
        new_node = new_fn[4]
        old_lines = {n for n, _ in removed}
        new_lines = {n for n, _, _ in added}
        if not same_local_binding_semantics(old_node, new_node, old_lines, new_lines):
            return None
        return self._diff_stmt_lists(path, old_node.body, new_node.body, ignore_ann)

    def _diff_stmt_lists(
        self,
        path: str,
        old_stmts: list[ast.stmt],
        new_stmts: list[ast.stmt],
        ignore_ann: bool,
    ) -> int | None:
        """Return coverage mask for the minimal changed statements/sub-blocks between two statement lists."""
        old_c = [canon_ast(s, ignore_ann) for s in old_stmts]
        new_c = [canon_ast(s, ignore_ann) for s in new_stmts]

        prefix = 0
        limit = min(len(old_c), len(new_c))
        while prefix < limit and old_c[prefix] == new_c[prefix]:
            prefix += 1

        suffix = 0
        rem_old = len(old_c) - prefix
        rem_new = len(new_c) - prefix
        while suffix < min(rem_old, rem_new) and old_c[len(old_c) - 1 - suffix] == new_c[len(new_c) - 1 - suffix]:
            suffix += 1

        mid_old = old_stmts[prefix : len(old_stmts) - suffix]
        mid_new = new_stmts[prefix : len(new_stmts) - suffix]

        if not mid_old and not mid_new:
            return 0

        # Pure insertion into this block: any test reaching the insertion point must pass through
        # an adjacent dominating non-docstring sibling statement in the same block.
        if not mid_old:
            sib = next((s for s in reversed(old_stmts[:prefix]) if not _is_docstring(s)), None)
            if sib is None:
                sib = next((s for s in old_stmts[prefix:] if not _is_docstring(s)), None)
            if sib is None:
                return None
            stop = header_end(sib) if isinstance(sib, COMPOUND_NODES) else sib.end_lineno
            return self._test_time_span(path, sib.lineno, stop)

        # Single compound statement edited in place: recurse into sub-blocks if its header is unchanged.
        if len(mid_old) == 1 and len(mid_new) == 1 and type(mid_old[0]) is type(mid_new[0]):
            so, sn = mid_old[0], mid_new[0]
            sub_pairs = _matching_compound_subblocks(so, sn, ignore_ann)
            if sub_pairs is not None:
                mask = 0
                collect_lines = self.impact.collect_lines.get(path, ())
                for idx, (bo, bn) in enumerate(sub_pairs):
                    sub_mask = self._diff_stmt_lists(path, bo, bn, ignore_ann)
                    if sub_mask is None:
                        if (
                            isinstance(so, (ast.If, ast.While))
                            and self.values is not None
                            and path.startswith("sqlglot/")
                        ):
                            hstop = header_end(so)
                            if not any(so.lineno <= ln <= hstop for ln in collect_lines):
                                try:
                                    vmask = self.values.narrow_branch(
                                        path, so.test, idx == 0, so, self.impact, header_stop=hstop
                                    )
                                except Exception:
                                    vmask = None
                                if vmask is not None:
                                    self._spans.append((path, so.lineno, hstop))
                                    sub_mask = vmask & self.impact.span(path, so.lineno, hstop)
                        if sub_mask is None:
                            return self._test_time_span(path, so.lineno, so.end_lineno)
                    mask |= sub_mask
                return mask
            # Only the compound header changed while all sub-blocks are identical.
            if isinstance(so, COMPOUND_NODES) and _same_compound_bodies(so, sn, ignore_ann):
                stop = header_end(so)
                if (
                    self.values is not None
                    and path.startswith("sqlglot/")
                    and isinstance(so, (ast.If, ast.While))
                    and not any(so.lineno <= ln <= stop for ln in self.impact.collect_lines.get(path, ()))
                ):
                    try:
                        vmask = self.values.narrow_expr(path, so.test, sn.test, so, "test", self.impact)
                    except Exception:
                        vmask = None
                    if vmask is not None:
                        self._spans.append((path, so.lineno, stop))
                        return vmask & self.impact.span(path, so.lineno, stop)
                return self._test_time_span(path, so.lineno, stop)
            # Single non-compound statement edited in place: try per-statement value-map narrowing.
            if (
                not isinstance(so, COMPOUND_NODES)
                and self.values is not None
                and path.startswith("sqlglot/")
                and not any(so.lineno <= ln <= so.end_lineno for ln in self.impact.collect_lines.get(path, ()))
            ):
                try:
                    vmask = self.values.narrow_stmt(path, so, sn, self.impact)
                except Exception:
                    vmask = None
                if vmask is not None:
                    self._spans.append((path, so.lineno, so.end_lineno))
                    return vmask & self.impact.span(path, so.lineno, so.end_lineno)

        mask = 0
        for s in mid_old:
            if _is_docstring(s):
                mask |= self.docs_mask
                continue
            span = self._test_time_span(path, s.lineno, s.end_lineno)
            if span is None:
                return None
            mask |= span
        return mask

    def runs_at_import(self, path: str, first: int, last: int) -> bool:
        lines = self.impact.import_lines.get(path)
        return bool(lines) and any(first <= n <= last for n in lines)

    def runs_before_tests(self, path: str, first: int, last: int) -> bool:
        """Runs while sqlglot or a test module imports. The verifier imports both once,
        before patching, so a hot-patched function like this would leave stale results."""
        if self.runs_at_import(path, first, last):
            return True
        lines = self.impact.collect_lines.get(path)
        return bool(lines) and any(first <= n <= last for n in lines)

    def _module_reach(self, path: str) -> int:
        impact = self.impact
        if self._no_pp_lazy:
            return impact.module_level_without_pp_lazy(path)
        full = impact.module_level(path)
        if full != impact.module_level_without_pp_lazy(path):
            self._hit_pp_lazy = True
        return full

    def function_level(self, path, module, body_start, end) -> int:
        mask = self.impact.span(path, body_start, end)
        if self.runs_before_tests(path, body_start, end):
            self._spans_ok = False
        else:
            self._spans.append((path, body_start, end))
        if self.runs_at_import(path, body_start, end):
            mask |= self._module_reach(path)
        return mask

    def container_level(self, path, module, scopes, lineno) -> int:
        """An import-time change at this line: its class's users & refs, or the whole module's reach."""
        self._spans_ok = False
        impact = self.impact
        cls = scopes.class_at(lineno)
        if cls is not None:
            head, end, qualname = cls
            usage = impact.class_usage(module, qualname)
            if usage is not None:
                return (
                    usage
                    | impact.class_ref_usage(module, qualname)
                    | impact.span(path, head, end)
                    | impact.by_file.get(path, 0)
                )
        return self._module_reach(path)

    def _safe_sig_edit(self, old_scopes: Scopes, new_scopes: Scopes, qualname: str) -> bool:
        """True when an existing function's header edit kept identical decorators and safe literal defaults."""
        old_rt = _runtime_def(old_scopes, qualname)
        new_rt = _runtime_def(new_scopes, qualname)
        if old_rt is None or new_rt is None:
            return False
        old_node, new_node = old_rt[4], new_rt[4]
        if [ast.dump(d) for d in old_node.decorator_list] != [ast.dump(d) for d in new_node.decorator_list]:
            return False
        if not (_has_safe_defaults(old_node) and _has_safe_defaults(new_node)):
            return False
        if not (old_scopes.has_future_annotations and new_scopes.has_future_annotations):
            if not (_has_no_annotations(old_node) and _has_no_annotations(new_node)):
                return False
        return True

    def old_line(self, path, module, scopes, new_scopes, lineno) -> int:
        function = scopes.function_at(lineno)
        if function is not None:
            body_start, end, head, qualname, node = function
            mask = self.function_level(path, module, body_start, end)
            if lineno < body_start:  # signature or decorator: callers and overriding both change
                if not self._safe_sig_edit(scopes, new_scopes, qualname):
                    mask |= self.container_level(path, module, scopes, head)
            if lineno in scopes.docstrings:
                mask |= self.docs_mask
            return mask
        span = scopes.lambda_at(lineno)
        if span is not None:
            mask = self.impact.span(path, span[0], span[1])
            if lineno == span[0]:
                mask |= self.container_level(path, module, scopes, lineno)
            elif self.runs_before_tests(path, span[0], span[1]):
                self._spans_ok = False
            else:
                self._spans.append((path, span[0], span[1]))
            return mask
        mask = self.container_level(path, module, scopes, lineno)
        if lineno in scopes.docstrings:
            mask |= self.docs_mask
        return mask

    def new_line(self, path, module, old_scopes, new_scopes, new_lineno, anchor) -> int:
        function = new_scopes.function_at(new_lineno)
        if function is not None:
            body_start, end, head, qualname, node = function
            spans = old_scopes.by_qualname.get(qualname)
            if spans and (new_lineno >= body_start or self._safe_sig_edit(old_scopes, new_scopes, qualname)):
                mask = 0
                for start, stop in spans:
                    mask |= self.function_level(path, module, start, stop)
                if new_lineno in new_scopes.docstrings:
                    mask |= self.docs_mask
                return mask
            # Newly added private helper function/method whose name never existed in pinned code
            # can only be reached via call sites added in the same patch (already covered in mask).
            if (
                not spans
                and self.impact.pinned_identifiers
                and node.name not in self.impact.pinned_identifiers
                and not node.name.endswith("_sql")
                and not (node.name.startswith("__") and node.name.endswith("__"))
                and all(_decorator_name(d) in SAFE_HOTPATCH_DECORATORS for d in node.decorator_list)
                and _has_safe_defaults(node)
                and (new_scopes.has_future_annotations or _has_no_annotations(node))
            ):
                return self.docs_mask if new_lineno in new_scopes.docstrings else 0
            # new function or changed signature: what defines it changed
            mask = self.container_level(path, module, new_scopes, head)
            for start, stop in spans or []:
                mask |= self.function_level(path, module, start, stop)
            return mask
        if new_scopes.lambda_at(new_lineno) is not None:
            for probe in (anchor - 1, anchor):
                span = old_scopes.lambda_at(probe)
                if span is not None and span[0] < probe <= span[1]:
                    if self.runs_before_tests(path, span[0], span[1]):
                        self._spans_ok = False
                    else:
                        self._spans.append((path, span[0], span[1]))
                    return self.impact.span(path, span[0], span[1])
        mask = self.container_level(path, module, new_scopes, new_lineno)
        if new_lineno in new_scopes.docstrings:
            mask |= self.docs_mask
        return mask
