"""Narrow condition-site and guarded-branch edits with the value map (map/values.bin).

Handles both:
  1. Condition sites (and/or/not/comparison): full rearrangements AND partial edits
     (modified/added/removed operands). Any modified/unmatched operand is compiled as
     a ("U",) unknown sentinel: tests whose recorded vectors short-circuit before
     reaching ("U",) — and produce the same truth, object identity (outside boolean
     context), and impure operand sequence — cannot behave differently.
  2. Ternary expressions (`body if test else orelse`) where `test` is an unchanged
     recorded logic site and only `body` or only `orelse` changed: tests where `test`
     always took the untouched branch cannot behave differently.
"""

from __future__ import annotations

import ast
import gzip
import marshal
from pathlib import Path

from blast_radius import valuesite as vs


class ValueMap:
    def __init__(self, path: Path, test_ids: list[str]):
        marshal_path = path.parent / "values.marshal"
        if marshal_path.is_file():
            raw = marshal.loads(marshal_path.read_bytes())
        else:
            raw = marshal.loads(gzip.decompress(path.read_bytes()))
        pos = {t: i for i, t in enumerate(test_ids)}
        self._remap = [pos.get(t) for t in raw["tests"]]
        self._identity = self._remap == list(range(len(test_ids)))
        self.files = raw["files"]
        self.imports = raw["import"]

    def _mask(self, mask: int) -> int:
        if self._identity:
            return mask
        out = 0
        i = 0
        while mask:
            if mask & 1 and self._remap[i] is not None:
                out |= 1 << self._remap[i]
            mask >>= 1
            i += 1
        return out

    def _eval_mutations(self, path: str, muts: list, impact) -> int | None:
        sites = self.files.get(path, {})
        imports = self.imports.get(path, {})
        acc = 0  # value-map test space
        extra = 0  # impact test space
        for mut in muts:
            kind = mut[0]
            if kind == "logic":
                _, old, new, parent, field = mut
                res = _narrow_logic_mut(old, new, parent, field, sites, imports)
                if res is None:
                    return None
                m_acc, m_rec = res
                acc |= m_acc
                extra |= impact.span(path, old.lineno, old.end_lineno) & ~self._mask(m_rec)
            elif kind == "ifexp":
                _, old_ifexp, test_node, need_truth = mut
                res = _narrow_ifexp_mut(test_node, need_truth, sites, imports)
                if res is None:
                    return None
                m_acc, m_rec = res
                acc |= m_acc
                extra |= impact.span(path, old_ifexp.lineno, old_ifexp.end_lineno) & ~self._mask(m_rec)
            else:
                return None
        return self._mask(acc) | extra

    def narrow(self, path: str, old_tree: ast.Module, new_tree: ast.Module, impact, hunks=None) -> int | None:
        muts = site_mutations(old_tree, new_tree, touched_old_lines(hunks) if hunks is not None else None)
        if muts is None or not muts:
            return None
        return self._eval_mutations(path, muts, impact)

    def narrow_stmt(self, path: str, old_stmt: ast.stmt, new_stmt: ast.stmt, impact) -> int | None:
        """Narrow a single paired statement edit inside a function body (supports multi-hunk files)."""
        muts = site_mutations(old_stmt, new_stmt, touched=None, start_depth=1)
        if muts is None or not muts:
            return None
        return self._eval_mutations(path, muts, impact)

    def narrow_expr(
        self,
        path: str,
        old_expr: ast.expr,
        new_expr: ast.expr,
        parent: ast.AST,
        field: str,
        impact,
    ) -> int | None:
        """Narrow a single paired expression (e.g. an If/While header test) inside a function body."""
        muts = site_mutations(old_expr, new_expr, touched=None, start_depth=1, root_parent=parent, root_field=field)
        if muts is None or not muts:
            return None
        return self._eval_mutations(path, muts, impact)

    def narrow_branch(
        self,
        path: str,
        test_node: ast.expr,
        need_truth: bool,
        stmt_node: ast.AST,
        impact,
        header_stop: int | None = None,
    ) -> int | None:
        """Keep only tests where unchanged logic condition `test_node` ever took truth value `need_truth`."""
        if not vs.is_logic(test_node):
            return None
        _, atoms_t, _ = vs.build(test_node)
        if not vs.instrumentable(atoms_t):
            return None
        res = _narrow_ifexp_mut(
            test_node,
            need_truth,
            self.files.get(path, {}),
            self.imports.get(path, {}),
        )
        if res is None:
            return None
        m_acc, m_rec = res
        stop = header_stop if header_stop is not None else stmt_node.end_lineno
        extra = impact.span(path, stmt_node.lineno, stop) & ~self._mask(m_rec)
        return self._mask(m_acc) | extra


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=False)


def _build_new_spec(new_node: ast.expr, atoms_o: list[ast.expr], impure: set[int]):
    """Compile `new_node` into a replay spec referencing `atoms_o` indices, using ('U',) for changed atoms."""
    # Exact-atoms fast path: preserves original 1-to-1 atom slots when only operators rearranged.
    spec_exact, atoms_n, _ = vs.build(new_node)
    if len(atoms_o) == len(atoms_n) and all(_dump(a) == _dump(b) for a, b in zip(atoms_o, atoms_n)):
        return spec_exact

    dump_to_old: dict[str, list[int]] = {}
    for i, a in enumerate(atoms_o):
        dump_to_old.setdefault(_dump(a), []).append(i)
    used_all: set[int] = set()
    used_impure: set[int] = set()

    def match_atom(node: ast.expr) -> int | None:
        cands = dump_to_old.get(_dump(node))
        if not cands:
            return None
        # Prefer an unused slot first so duplicate pure expressions keep their distinct recorded slots.
        for ci in cands:
            if ci not in used_all:
                used_all.add(ci)
                if ci in impure:
                    used_impure.add(ci)
                return ci
        for ci in cands:
            if ci not in impure and ci not in used_impure:
                return ci
        return None

    def walk(node: ast.expr):
        if isinstance(node, ast.BoolOp):
            return ("&" if isinstance(node.op, ast.And) else "|", tuple(walk(v) for v in node.values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child = walk(node.operand)
            return ("U",) if child == ("U",) else ("N", child)
        if isinstance(node, ast.Compare):
            ops = tuple(type(op).__name__ for op in node.ops)
            operands = (node.left, *node.comparators)
            idxs: list[int] = []
            for opnd in operands:
                ci = match_atom(opnd)
                if ci is None:
                    return ("U",)
                idxs.append(ci)
            return ("C", tuple(idxs), ops, tuple(range(len(ops))))
        ci = match_atom(node)
        return ("U",) if ci is None else ("A", ci, -1)

    return walk(new_node)


def _replay_new(spec, vec: str, truth: dict[int, int], links: dict[tuple[int, int], int], offs: dict[int, int]):
    """Replay a generalized spec (which may contain ('U',) nodes) over a recorded outcome vector."""
    seq: list[int] = []

    def tslot(i: int) -> str:
        sl = truth.get(i)
        if sl is None:
            raise vs.Unknown
        c = vec[offs[sl]]
        if c not in "TFtf":
            raise vs.Unknown
        return c

    def ev(s):
        kind = s[0]
        if kind == "U":
            raise vs.Unknown
        if kind == "A":
            c = tslot(s[1])
            seq.append(s[1])
            return c in "Tt", ("b",) if c in "TF" else ("a", s[1])
        if kind == "N":
            v, _ = ev(s[1])
            return (not v), ("b",)
        if kind in "&|":
            for c in s[1]:
                v, ident = ev(c)
                if (kind == "&") != v:
                    return v, ident
            return v, ident
        atoms, ops = s[1], s[2]
        seq.append(atoms[0])
        for j, op in enumerate(ops):
            seq.append(atoms[j + 1])
            sl = links.get((atoms[j], atoms[j + 1]))
            if sl is None:
                raise vs.Unknown
            c = vec[offs[sl] + vs.REL_INDEX[op]]
            if c not in "TFtf":
                raise vs.Unknown
            v = c in "Tt"
            ident = ("b",) if c in "TF" else ("c", sl, op)
            if not v and j < len(ops) - 1:
                return v, ident
        return v, ident

    v, ident = ev(spec)
    return v, ident, seq


def _narrow_logic_mut(
    old: ast.expr,
    new: ast.expr,
    parent: ast.AST | None,
    field: str | None,
    sites: dict,
    imports: dict,
) -> tuple[int, int] | None:
    spec_o, atoms_o, nslots_o = vs.build(old)
    truth, links = vs.slot_maps(spec_o)
    offs = vs.offsets(spec_o, nslots_o)
    impure = {i for i, a in enumerate(atoms_o) if not vs.is_pure(a)}
    spec_n = _build_new_spec(new, atoms_o, impure)
    if spec_n == ("U",):
        return None
    boolctx = vs.bool_context(parent, field)
    key = vs.site_key(old)

    def differs(vec: str) -> bool:
        try:
            vo, io, so = vs.replay(spec_o, vec, truth, links, offs)
            vn, in_, sn = _replay_new(spec_n, vec, truth, links, offs)
        except vs.Unknown:
            return True
        if vo != vn:
            return True
        if not boolctx and io != in_ and not (io == ("b",) and in_ == ("b",)):
            return True
        return [a for a in so if a in impure] != [a for a in sn if a in impure]

    if any(differs(v) for v in imports.get(key, ())):
        return None
    acc = 0
    recorded = 0
    for vec, mask in sites.get(key, ()):
        recorded |= mask
        if differs(vec):
            acc |= mask
    return acc, recorded


def _narrow_ifexp_mut(
    test_node: ast.expr,
    need_truth: bool,
    sites: dict,
    imports: dict,
) -> tuple[int, int] | None:
    """Keep only tests where `test_node` evaluated to `need_truth` (or raised/Unknown)."""
    spec_o, _, nslots_o = vs.build(test_node)
    truth, links = vs.slot_maps(spec_o)
    offs = vs.offsets(spec_o, nslots_o)
    key = vs.site_key(test_node)

    def reached(vec: str) -> bool:
        try:
            vo, _, _ = vs.replay(spec_o, vec, truth, links, offs)
        except vs.Unknown:
            return True
        return vo == need_truth

    if any(reached(v) for v in imports.get(key, ())):
        return None
    acc = 0
    recorded = 0
    for vec, mask in sites.get(key, ()):
        recorded |= mask
        if reached(vec):
            acc |= mask
    return acc, recorded


def touched_old_lines(hunks) -> set[int]:
    """Old lines a hunk removes, plus the old lines on both sides of every insertion."""
    out: set[int] = set()
    for old_start, _, _, body in hunks:
        ln = old_start
        for line in body:
            tag = line[:1]
            if tag == "-":
                out.add(ln)
                ln += 1
            elif tag == "+":
                out.update((ln - 1, ln))
            else:
                ln += 1
    return out


def site_mutations(
    old_tree: ast.AST,
    new_tree: ast.AST,
    touched: set[int] | None = None,
    start_depth: int = 0,
    root_parent: ast.AST | None = None,
    root_field: str | None = None,
):
    """Return mutation descriptors when every difference is a value-map-narrowable site edit, else None.
    With `touched` (old line numbers the patch changed), nodes wholly outside it are taken as equal."""
    muts: list = []

    def same(o, n, depth: int, parent, field) -> bool:
        if touched is not None and isinstance(o, (ast.stmt, ast.expr)):
            lo, hi = o.lineno, o.end_lineno
            if type(o) is type(n) and not any(lo <= ln <= hi for ln in touched):
                return True
        if isinstance(o, ast.AST) and vs.is_logic(o):
            if isinstance(n, ast.AST) and _dump(o) == _dump(n):
                return True
            if depth == 0 or not isinstance(n, ast.expr):
                return False
            _, atoms_o, _ = vs.build(o)
            if not vs.instrumentable(atoms_o):
                return False
            muts.append(("logic", o, n, parent, field))
            return True
        if isinstance(o, ast.IfExp) and isinstance(n, ast.IfExp) and depth > 0 and vs.is_logic(o.test):
            if _dump(o.test) == _dump(n.test):
                _, atoms_t, _ = vs.build(o.test)
                if vs.instrumentable(atoms_t):
                    body_same = _dump(o.body) == _dump(n.body)
                    else_same = _dump(o.orelse) == _dump(n.orelse)
                    if body_same and else_same:
                        return True
                    if body_same and not else_same:
                        muts.append(("ifexp", o, o.test, False))
                        return True
                    if else_same and not body_same:
                        muts.append(("ifexp", o, o.test, True))
                        return True
        if type(o) is not type(n):
            return False
        if isinstance(o, ast.AST):
            if (
                isinstance(o, (ast.stmt, ast.expr))
                and not isinstance(o, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and (o.end_lineno - o.lineno) <= 5
                and _dump(o) == _dump(n)
            ):
                return True
            inner = depth + 1 if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else depth
            for name, vo in ast.iter_fields(o):
                vn = getattr(n, name, None)
                if isinstance(vo, list):
                    if not isinstance(vn, list) or len(vo) != len(vn):
                        return False
                    for a, b in zip(vo, vn):
                        if isinstance(a, ast.AST):
                            if not same(a, b, inner, o, name):
                                return False
                        elif a != b:
                            return False
                elif isinstance(vo, ast.AST):
                    if not isinstance(vn, ast.AST) or not same(vo, vn, inner, o, name):
                        return False
                elif vo != vn:
                    return False
            return True
        return o == n

    return muts if same(old_tree, new_tree, start_depth, root_parent, root_field) else None
