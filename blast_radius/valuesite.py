"""Condition sites: the boolean/comparison expressions the value map records.

Shared by the map-building plugin (tools/value_plugin.py), which records, per test,
the outcome vectors each site produced at the pinned commit, and by the selector,
which replays those vectors through a mutated site to find the tests whose
evaluation of it would change.

A site is a maximal tree of `and`/`or`/`not`/comparison nodes inside a function
body. Its leaves ("atoms") are the other sub-expressions. Each truth atom (an
operand of and/or/not) and each comparison link gets one slot in a vector:

* truth slot: T/F (exact bool), t/f (other truthy/falsy object), E (truth test or
  speculative evaluation raised), X (raised while the program evaluated it),
  - (not evaluated: an impure site short-circuited it).
* comparison slot: one char per relation in RELS (same alphabet, ? = not computed).
"""

from __future__ import annotations

import ast

RELS = ("Is", "IsNot", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn")
REL_INDEX = {name: i for i, name in enumerate(RELS)}

PURE_FUNCS = frozenset(
    {
        "isinstance", "issubclass", "len", "type", "callable", "bool", "id", "getattr", "hasattr",
        "tuple", "frozenset", "set", "list", "dict", "str", "int", "float", "repr", "any", "all",
        "min", "max", "sum", "sorted", "reversed", "enumerate", "zip", "abs", "ord", "chr", "seq_get",
        "ensure_list", "ensure_collection",
    }
)
PURE_METHODS = frozenset(
    {
        "get", "startswith", "endswith", "upper", "lower", "isupper", "islower", "isdigit", "isalpha",
        "isalnum", "isidentifier", "isspace", "strip", "lstrip", "rstrip", "text", "is_type", "keys",
        "values", "items", "count", "issubset", "issuperset", "isdisjoint", "arg_key", "find_ancestor",
        "copy", "meta_get", "find", "find_all", "unnest", "sql_name", "sql_names", "to_py",
    }
)
# Properties that lazily cache derived state: evaluating them early could change later behavior.
IMPURE_ATTRS = frozenset(
    {
        "dag", "supported_table_args", "meta", "all_columns", "select_aliases", "columns",
        "selected_sources", "references", "external_columns", "local_columns", "pivots",
    }
)
_PURE_NODES = (
    ast.Name, ast.Constant, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Set, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.JoinedStr, ast.FormattedValue,
    ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.keyword, ast.comprehension,
    ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.Lambda, ast.arguments, ast.arg,
    ast.Starred,
)
_FORBIDDEN = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)


def is_logic(node: ast.AST) -> bool:
    return isinstance(node, (ast.BoolOp, ast.Compare)) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    )


def site_key(node: ast.AST) -> tuple[int, int, int, int]:
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def build(node: ast.expr):
    """(spec, atoms, nslots). spec nodes:
    ('A', atom, slot) | ('N', child) | ('&', children) | ('|', children) | ('C', atoms, ops, slots)."""
    atoms: list[ast.expr] = []
    counter = [0]

    def slot() -> int:
        counter[0] += 1
        return counter[0] - 1

    def walk(n: ast.expr):
        if isinstance(n, ast.BoolOp):
            return ("&" if isinstance(n.op, ast.And) else "|", tuple(walk(v) for v in n.values))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return ("N", walk(n.operand))
        if isinstance(n, ast.Compare):
            idx = []
            for operand in (n.left, *n.comparators):
                atoms.append(operand)
                idx.append(len(atoms) - 1)
            return ("C", tuple(idx), tuple(type(op).__name__ for op in n.ops), tuple(slot() for _ in n.ops))
        atoms.append(n)
        return ("A", len(atoms) - 1, slot())

    spec = walk(node)
    return spec, atoms, counter[0]


def instrumentable(atoms: list[ast.expr]) -> bool:
    for atom in atoms:
        for sub in ast.walk(atom):
            if isinstance(sub, _FORBIDDEN):
                return False
            if isinstance(sub, ast.Name) and sub.id in ("super", "__class__", "locals", "vars"):
                return False
    return True


def is_pure(atom: ast.expr) -> bool:
    for sub in ast.walk(atom):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in PURE_FUNCS:
                continue
            if isinstance(f, ast.Attribute) and f.attr in PURE_METHODS:
                continue
            return False
        if isinstance(sub, ast.Attribute):
            if sub.attr in IMPURE_ATTRS:
                return False
            continue
        if not isinstance(sub, _PURE_NODES):
            return False
    return True


def is_bare_test(node: ast.AST, parent, field) -> bool:
    """True for non-logic test expressions on If/While/IfExp whose truthiness controls execution."""
    return (
        isinstance(node, ast.expr)
        and not is_logic(node)
        and field == "test"
        and isinstance(parent, (ast.If, ast.While, ast.IfExp))
    )


def is_recorded_site(node: ast.AST, parent, field) -> bool:
    """True if node is instrumented as a condition site (either logic tree or bare If/While/IfExp test)."""
    return isinstance(node, ast.expr) and (is_logic(node) or is_bare_test(node, parent, field))


def iter_sites(tree: ast.AST):
    """Yield (root, parent, field, depth) for every condition site, outermost first.
    depth counts enclosing function/lambda scopes (0: module or class body)."""

    def visit(node: ast.AST, depth: int, parent, field):
        if is_logic(node):
            yield node, parent, field, depth
            spec, atoms, _ = build(node)
            for atom in atoms:
                yield from visit(atom, depth, node, "atom")
            return
        if is_bare_test(node, parent, field):
            yield node, parent, field, depth
        inner = depth + 1 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else depth
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        yield from visit(item, inner, node, name)
            elif isinstance(value, ast.AST):
                yield from visit(value, inner, node, name)

    yield from visit(tree, 0, None, None)


def bool_context(parent: ast.AST | None, field: str | None) -> bool:
    """Only the truth of the value matters where it is used."""
    if parent is None:
        return False
    if field == "test" and isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)):
        return True
    if field == "ifs" and isinstance(parent, ast.comprehension):
        return True
    if (
        field == "args"
        and isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id == "bool"
    ):
        return True
    return False


# ---------------------------------------------------------------- replay (selector side)


class Unknown(Exception):
    pass


def slot_maps(spec):
    """Original spec -> ({atom: truth slot}, {(left atom, right atom): link slot})."""
    truth: dict[int, int] = {}
    links: dict[tuple[int, int], int] = {}

    def walk(s):
        kind = s[0]
        if kind == "A":
            truth[s[1]] = s[2]
        elif kind == "N":
            walk(s[1])
        elif kind in "&|":
            for c in s[1]:
                walk(c)
        else:
            for j, sl in enumerate(s[3]):
                links[(s[1][j], s[1][j + 1])] = sl

    walk(spec)
    return truth, links


def replay(spec, vec: str, truth: dict[int, int], links: dict[tuple[int, int], int], nlinks_chars: dict[int, int]):
    """Evaluate spec over a recorded vector: (truth, identity, evaluated atoms). Raises Unknown."""
    seq: list[int] = []

    def tslot(i: int) -> str:
        sl = truth.get(i)
        if sl is None:
            raise Unknown
        c = vec[nlinks_chars[sl]]
        if c not in "TFtf":
            raise Unknown
        return c

    def ev(s):
        kind = s[0]
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
                raise Unknown
            c = vec[nlinks_chars[sl] + REL_INDEX[op]]
            if c not in "TFtf":
                raise Unknown
            v = c in "Tt"
            ident = ("b",) if c in "TF" else ("c", sl, op)
            if not v and j < len(ops) - 1:
                return v, ident
        return v, ident

    v, ident = ev(spec)
    return v, ident, seq


def offsets(spec, nslots: int) -> dict[int, int]:
    """Char offset of each slot in a vector: truth slots take 1 char, link slots len(RELS)."""
    width: dict[int, int] = {}

    def walk(s):
        kind = s[0]
        if kind == "A":
            width[s[2]] = 1
        elif kind == "N":
            walk(s[1])
        elif kind in "&|":
            for c in s[1]:
                walk(c)
        else:
            for sl in s[3]:
                width[sl] = len(RELS)

    walk(spec)
    out, pos = {}, 0
    for sl in range(nslots):
        out[sl] = pos
        pos += width[sl]
    return out
