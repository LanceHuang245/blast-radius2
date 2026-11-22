"""Static Python analysis: scopes, statements, binding shape, dict entries, comment-only checks."""

from __future__ import annotations

import ast
import io
import re
import tokenize

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
COMPOUND_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Match,
)
if hasattr(ast, "TryStar"):
    COMPOUND_NODES += (ast.TryStar,)

ENCODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


_LEAF_EXPR_NODES = (
    ast.Name,
    ast.Constant,
    ast.expr_context,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.arg,
)


class Scopes:
    """Where each line of a module sits: functions, classes, statements, class assigns."""

    def __init__(self, tree: ast.Module, has_lambda: bool = True, text: str | None = None):
        self.tree = tree
        self._has_lambda = has_lambda
        # Lines holding the word `lambda`: expressions spanning none of them hold no lambda, so
        # the walk need not descend into them (most of a big module's nodes).
        self._lambda_lines: list[int] | None = (
            [i for i, line in enumerate(text.splitlines(), 1) if "lambda" in line] if text is not None else None
        )
        self.functions: list[tuple[int, int, int, str, ast.AST]] = []
        self.lambdas: list[tuple[int, int, int]] = []
        self.classes: list[tuple[int, int, str]] = []
        self.by_qualname: dict[str, list[tuple[int, int]]] = {}
        self.statements: list[tuple[int, int, ast.stmt]] = []
        self.class_assigns: dict[tuple[str, str], ast.stmt] = {}
        self.module_assigns: dict[str, ast.stmt] = {}
        self.docstrings: set[int] = set()
        self.type_checking_ranges: list[tuple[int, int]] = []
        self.has_future_annotations = any(
            isinstance(s, ast.ImportFrom)
            and s.module == "__future__"
            and any(a.name == "annotations" for a in s.names)
            for s in tree.body
        )
        for stmt in tree.body:
            attr = _single_assign_target(stmt)
            if attr is not None:
                self.module_assigns[attr] = stmt
        self._walk(tree, "")

    def _walk(self, node: ast.AST, prefix: str) -> None:
        if isinstance(node, _LEAF_EXPR_NODES):
            return
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and _is_docstring(body[0]):
            self.docstrings.update(range(body[0].lineno, body[0].end_lineno + 1))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _LEAF_EXPR_NODES):
                continue
            if isinstance(child, ast.stmt):
                self.statements.append((child.lineno, child.end_lineno, child))
                if isinstance(child, ast.If) and is_type_checking_guard(child) and child.body:
                    self.type_checking_ranges.append((child.body[0].lineno, child.body[-1].end_lineno))
                if isinstance(child, FUNCTION_NODES):
                    qualname = prefix + child.name
                    head = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    body_start = child.body[0].lineno
                    self.functions.append((body_start, child.end_lineno, head, qualname, child))
                    self.by_qualname.setdefault(qualname, []).append((body_start, child.end_lineno))
                    self._walk(child, qualname + ".<locals>.")
                elif isinstance(child, ast.ClassDef):
                    qualname = prefix + child.name
                    head = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    self.classes.append((head, child.end_lineno, qualname))
                    for stmt in child.body:
                        attr = _single_assign_target(stmt)
                        if attr is not None:
                            self.class_assigns[(qualname, attr)] = stmt
                    self._walk(child, qualname + ".")
                else:
                    self._walk(child, prefix)
            elif self._has_lambda and (
                isinstance(child, (ast.ExceptHandler, ast.match_case)) or self._may_hold_lambda(child)
            ):
                if isinstance(child, ast.Lambda):
                    # (body start, body end, header line): a body line below the header only runs
                    # when the lambda is called; the header line also runs where it is defined.
                    self.lambdas.append((child.body.lineno, child.body.end_lineno, child.lineno))
                self._walk(child, prefix)
            elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                self._walk(child, prefix)

    def _may_hold_lambda(self, node: ast.AST) -> bool:
        lines = self._lambda_lines
        if lines is None:
            return True
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            return True
        lo, hi = 0, len(lines)
        while lo < hi:  # first lambda line >= start
            mid = (lo + hi) // 2
            if lines[mid] < start:
                lo = mid + 1
            else:
                hi = mid
        return lo < len(lines) and lines[lo] <= end

    def function_at(self, lineno: int):
        """Innermost function whose body or header holds the line."""
        best = None
        for body_start, end, head, qualname, node in self.functions:
            if head <= lineno <= end and (best is None or head >= best[2]):
                best = (body_start, end, head, qualname, node)
        return best

    def lambda_at(self, lineno: int):
        best = None
        for start, end, head in self.lambdas:
            if start <= lineno <= end and (best is None or start >= best[0]):
                best = (start, end, head)
        return best

    def class_at(self, lineno: int):
        best = None
        for head, end, qualname in self.classes:
            if head <= lineno <= end and (best is None or head >= best[0]):
                best = (head, end, qualname)
        return best

    def statement_at(self, lineno: int):
        best = None
        for start, end, node in self.statements:
            if start <= lineno <= end and (best is None or start >= best[0]):
                best = (start, end, node)
        return best

    def class_assign_at(self, lineno: int) -> tuple[str, str, ast.stmt] | None:
        """If lineno lies inside a direct class-body attribute assignment, return (cls, attr, stmt)."""
        cls = self.class_at(lineno)
        if cls is None:
            return None
        cls_qualname = cls[2]
        for (cqual, attr), stmt in self.class_assigns.items():
            if cqual == cls_qualname and stmt.lineno <= lineno <= stmt.end_lineno:
                return cls_qualname, attr, stmt
        return None

    def module_assign_at(self, lineno: int) -> tuple[str, ast.stmt] | None:
        """If lineno lies inside a top-level module assignment, return (attr, stmt)."""
        if self.class_at(lineno) is not None or self.function_at(lineno) is not None:
            return None
        for attr, stmt in self.module_assigns.items():
            if stmt.lineno <= lineno <= stmt.end_lineno:
                return attr, stmt
        return None

    def in_type_checking(self, lineno: int) -> bool:
        return any(s <= lineno <= e for s, e in self.type_checking_ranges)


def is_type_checking_guard(node: ast.If) -> bool:
    """True for `if TYPE_CHECKING:` or `if t.TYPE_CHECKING:` / `if typing.TYPE_CHECKING:`."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def canon_ast(node: ast.AST | list | None, ignore_annotations: bool = False):
    """Canonical structural tuple of an AST node for semantics-preserving equality checks.

    Ignores formatting, quote style, parens, comments, `if TYPE_CHECKING:` bodies,
    `return None` vs bare `return`, and (when `from __future__ import annotations`
    is active and safe) type annotations. Tags constants with their exact Python type
    so `1` never compares equal to `True`.
    """
    if node is None:
        return None
    if isinstance(node, list):
        out = []
        for item in node:
            if isinstance(item, ast.If) and is_type_checking_guard(item):
                if item.orelse:
                    out.extend(canon_ast(item.orelse, ignore_annotations))
                continue
            out.append(canon_ast(item, ignore_annotations))
        return tuple(out)
    if not isinstance(node, ast.AST):
        return (type(node).__name__, node)
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value).__name__, node.value)
    if isinstance(node, ast.Return):
        val = node.value
        if isinstance(val, ast.Constant) and val.value is None:
            return ("Return", None)
        return ("Return", canon_ast(val, ignore_annotations))
    if ignore_annotations:
        if isinstance(node, ast.arg):
            return ("arg", node.arg)
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return ("Assign", (canon_ast(node.target, True),), canon_ast(node.value, True))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return (
                type(node).__name__,
                node.name,
                canon_ast(node.args, True),
                canon_ast(node.body, True),
                canon_ast(node.decorator_list, True),
            )
    fields = []
    for name in node._fields:
        if name == "type_comment":
            continue
        if ignore_annotations and isinstance(node, ast.AnnAssign) and name == "annotation":
            continue
        fields.append(canon_ast(getattr(node, name), ignore_annotations))
    return (type(node).__name__, *fields)


def _same_stmt_block(
    old_body: list[ast.stmt],
    new_body: list[ast.stmt],
    ignore_annotations: bool,
    old_lines: set[int],
    new_lines: set[int],
) -> bool | None:
    """Compare two statement lists by only canonicalizing statements intersecting changed lines."""
    old_stmts = [
        s for s in old_body
        if not (isinstance(s, ast.If) and is_type_checking_guard(s) and not s.orelse)
    ]
    new_stmts = [
        s for s in new_body
        if not (isinstance(s, ast.If) and is_type_checking_guard(s) and not s.orelse)
    ]
    if len(old_stmts) != len(new_stmts):
        return None
    for so, sn in zip(old_stmts, new_stmts):
        touched = any(so.lineno <= ln <= so.end_lineno for ln in old_lines) or any(
            sn.lineno <= ln <= sn.end_lineno for ln in new_lines
        )
        if not touched:
            continue
        if isinstance(so, ast.ClassDef) and isinstance(sn, ast.ClassDef):
            if (
                so.name == sn.name
                and canon_ast(so.bases, ignore_annotations) == canon_ast(sn.bases, ignore_annotations)
                and canon_ast(so.keywords, ignore_annotations) == canon_ast(sn.keywords, ignore_annotations)
                and canon_ast(so.decorator_list, ignore_annotations)
                == canon_ast(sn.decorator_list, ignore_annotations)
            ):
                sub = _same_stmt_block(so.body, sn.body, ignore_annotations, old_lines, new_lines)
                if sub is not None:
                    if not sub:
                        return False
                    continue
        if canon_ast(so, ignore_annotations) != canon_ast(sn, ignore_annotations):
            return False
    return True


def same_ast(
    old_tree: ast.Module,
    new_tree: ast.Module,
    old_text: str,
    new_text: str,
    ignore_annotations: bool = False,
    old_lines: set[int] | None = None,
    new_lines: set[int] | None = None,
) -> bool:
    """True when two modules have identical normalized ASTs and identical encoding cookies.

    When changed line sets are provided, only statements/methods intersecting the edit
    are canonicalized, avoiding full-file AST traversal on multi-thousand-line modules.
    """
    if _encoding_cookie(old_text) != _encoding_cookie(new_text):
        return False
    if old_lines is not None and new_lines is not None:
        fast = _same_stmt_block(old_tree.body, new_tree.body, ignore_annotations, old_lines, new_lines)
        if fast is not None:
            return fast
    return canon_ast(old_tree, ignore_annotations) == canon_ast(new_tree, ignore_annotations)


def _single_assign_target(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
        return stmt.target.id
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def header_end(node: ast.stmt) -> int:
    """Last line of a compound statement's header (before its first body line)."""
    first_body = min(
        (
            child.lineno
            for field in ("body", "orelse", "handlers", "finalbody", "cases")
            for child in getattr(node, field, []) or []
            if hasattr(child, "lineno")
        ),
        default=node.end_lineno + 1,
    )
    return max(node.lineno, first_body - 1)


def _function_binding_shape(func_node: ast.AST) -> tuple:
    """Capture lexical scope bindings and generator/coroutine flags of a function body.

    An edit in an unexecuted branch that adds a local assignment, `global`/`nonlocal`,
    or `yield` changes the whole function's semantics at compile time.
    """
    locals_bound: set[str] = set()
    globals_decl: set[str] = set()
    nonlocals_decl: set[str] = set()
    has_yield = False

    def collect_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            locals_bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect_target(elt)
        elif isinstance(target, ast.Starred):
            collect_target(target.value)

    stack: list[ast.AST] = list(getattr(func_node, "body", []))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locals_bound.add(cur.name)
            continue
        if isinstance(cur, ast.Lambda):
            continue
        if isinstance(cur, (ast.Yield, ast.YieldFrom)):
            has_yield = True
        elif isinstance(cur, ast.Global):
            globals_decl.update(cur.names)
        elif isinstance(cur, ast.Nonlocal):
            nonlocals_decl.update(cur.names)
        elif isinstance(cur, ast.Name) and isinstance(cur.ctx, (ast.Store, ast.Del)):
            locals_bound.add(cur.id)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            locals_bound.add(cur.name)
        elif isinstance(cur, (ast.Import, ast.ImportFrom)):
            for alias in cur.names:
                locals_bound.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(cur, ast.NamedExpr):
            collect_target(cur.target)
        stack.extend(ast.iter_child_nodes(cur))

    return (
        isinstance(func_node, ast.AsyncFunctionDef),
        has_yield,
        frozenset(locals_bound),
        frozenset(globals_decl),
        frozenset(nonlocals_decl),
    )


def _names_used_outside_lines(func_node: ast.AST, candidates: frozenset[str], edited_lines: set[int]) -> bool:
    """True if any variable in `candidates` is referenced outside `edited_lines` or inside a nested scope."""
    for arg in getattr(func_node, "args", ast.arguments()).posonlyargs + getattr(
        func_node, "args", ast.arguments()
    ).args + getattr(func_node, "args", ast.arguments()).kwonlyargs:
        if arg.arg in candidates:
            return True
    for vararg in (getattr(func_node.args, "vararg", None), getattr(func_node.args, "kwarg", None)):
        if vararg is not None and vararg.arg in candidates:
            return True

    stack: list[tuple[ast.AST, bool]] = [(s, False) for s in getattr(func_node, "body", [])]
    while stack:
        cur, in_nested = stack.pop()
        nested = in_nested or isinstance(
            cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        )
        if isinstance(cur, ast.Name) and cur.id in candidates:
            if nested or getattr(cur, "lineno", -1) not in edited_lines:
                return True
        elif isinstance(cur, (ast.Global, ast.Nonlocal)) and any(n in candidates for n in cur.names):
            return True
        stack.extend((ch, nested) for ch in ast.iter_child_nodes(cur))
    return False


def same_local_binding_semantics(
    old_func: ast.AST,
    new_func: ast.AST,
    old_lines: set[int],
    new_lines: set[int],
) -> bool:
    """True when an edit inside a function cannot alter lexical bindings on unexecuted paths."""
    old_async, old_yield, old_locals, old_globals, old_nonlocals = _function_binding_shape(old_func)
    new_async, new_yield, new_locals, new_globals, new_nonlocals = _function_binding_shape(new_func)
    if (
        old_async != new_async
        or old_yield != new_yield
        or old_globals != new_globals
        or old_nonlocals != new_nonlocals
    ):
        return False
    diff_locals = old_locals ^ new_locals
    if not diff_locals:
        return True
    return not _names_used_outside_lines(old_func, diff_locals, old_lines) and not _names_used_outside_lines(
        new_func, diff_locals, new_lines
    )


def _ast_dict_key_tokens(key_node: ast.AST) -> tuple[str, ...] | None:
    """Convert an AST dict key node into candidate _TrackedDict lookup tokens."""
    if isinstance(key_node, ast.Constant):
        if isinstance(key_node.value, str):
            return ("s:" + key_node.value,)
        if isinstance(key_node.value, int) and not isinstance(key_node.value, bool):
            return (f"i:{key_node.value}",)
        return None
    if isinstance(key_node, ast.Attribute):
        parent = (
            key_node.value.id
            if isinstance(key_node.value, ast.Name)
            else key_node.value.attr if isinstance(key_node.value, ast.Attribute) else None
        )
        if parent:
            return (f"e:{parent}.{key_node.attr}", "t:" + key_node.attr)
        return ("t:" + key_node.attr,)
    return None


def _extract_dict_entries(dict_node: ast.AST) -> tuple[dict[tuple[str, ...], ast.AST], list[str]] | None:
    """Extract statically known key -> value AST pairs and spread expressions from an ast.Dict."""
    if not isinstance(dict_node, ast.Dict):
        return None
    entries: dict[tuple[str, ...], ast.AST] = {}
    spreads: list[str] = []
    for k, v in zip(dict_node.keys, dict_node.values):
        if k is None:
            if (
                isinstance(v, ast.DictComp)
                and len(v.generators) == 1
                and not v.generators[0].ifs
                and not v.generators[0].is_async
                and isinstance(v.generators[0].target, ast.Name)
                and isinstance(v.key, ast.Name)
                and v.key.id == v.generators[0].target.id
                and isinstance(v.generators[0].iter, (ast.Set, ast.List, ast.Tuple))
            ):
                for elt in v.generators[0].iter.elts:
                    toks = _ast_dict_key_tokens(elt)
                    if toks is None or toks in entries:
                        return None
                    entries[toks] = v.value
                continue
            spreads.append(ast.dump(v))
            continue
        toks = _ast_dict_key_tokens(k)
        if toks is None or toks in entries:
            return None
        entries[toks] = v
    return entries, spreads


def _encoding_cookie(text: str) -> str | None:
    for line in text.split("\n", 2)[:2]:
        match = ENCODING_RE.match(line)
        if match:
            return match.group(1).lower()
    return None


def same_code(old_text: str, new_text: str) -> bool:
    """True when the two sources differ only in comments and blank lines (and same encoding cookie)."""
    if _encoding_cookie(old_text) != _encoding_cookie(new_text):
        return False

    def significant(text):
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL):
                continue
            tokens.append((token.type, token.string))
        return tokens

    try:
        return significant(old_text) == significant(new_text)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return False


def comment_lines_only(lines: list[str]) -> bool:
    return all(not line.strip() or line.strip().startswith("#") for line in lines)


def indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))
