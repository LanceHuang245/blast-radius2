"""Import-time state of sqlglot modules: who runs which lines at import, and a digest of what it built.

Some sqlglot modules parse SQL while importing (generators/duckdb.py builds expression constants
with exp.maybe_parse), so parser and tokenizer statements run before any test. An edit to such a
statement can only change the tests' world through the state those imports leave behind. This
module records, at build time, which module's import ("owner") runs each line, and a digest of
each owner's data; at run time the patched owners are imported again and their digests compared.

Build: python -m blast_radius.import_state   (writes map/import_state.marshal)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import marshal
import os
import subprocess
import sys
import types
from collections import defaultdict

from blast_radius import APP, SOURCE

STATE_FILE = APP / "map" / "import_state.marshal"


class Undigestable(Exception):
    """A value whose content cannot be compared deterministically."""


def _digest_value(value, owner: str, seen: set[int], out: list[str]) -> None:
    from sqlglot import exp
    from sqlglot.serde import dump

    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        out.append(repr(value))
    elif isinstance(value, exp.Expr):
        out.append("E" + repr(dump(value)))
    elif isinstance(value, (tuple, list)):
        out.append(type(value).__name__ + "[")
        for item in value:
            _digest_value(item, owner, seen, out)
        out.append("]")
    elif isinstance(value, (set, frozenset)):
        # Iteration order follows hashes: digest each element on its own copy of `seen`, so what
        # one element marks as seen cannot change how another is written.
        parts = []
        for item in value:
            sub: list[str] = []
            _digest_value(item, owner, set(seen), sub)
            parts.append("".join(sub))
        for item in value:
            if isinstance(item, (type, types.FunctionType)) or type(item).__module__.startswith("sqlglot"):
                seen.add(id(item))
        out.append("{" + ",".join(sorted(parts)) + "}")
    elif isinstance(value, dict):
        # Visit values in key order, not insertion order: `seen` makes a shared object's first
        # visit write more than later ones, so the visiting order must not depend on how
        # (or in what order of imports) the dict was filled.
        keyed = []
        for key, item in value.items():
            ksub: list[str] = []
            _digest_value(key, owner, set(seen), ksub)
            keyed.append(("".join(ksub), key, item))
        keyed.sort(key=lambda entry: entry[0])
        parts = []
        for key_text, _key, item in keyed:
            sub: list[str] = [key_text, ":"]
            _digest_value(item, owner, seen, sub)
            parts.append("".join(sub))
        out.append("{" + ",".join(parts) + "}")
    elif isinstance(value, (staticmethod, classmethod)):
        _digest_value(value.__func__, owner, seen, out)
    elif isinstance(value, property):
        for fn in (value.fget, value.fset, value.fdel):
            _digest_value(fn, owner, seen, out)
    elif isinstance(value, types.FunctionType):
        # Code is unaffected by import-time parsing; what import built is in defaults and closures.
        out.append("F" + value.__qualname__)
        if id(value) in seen:
            return
        seen.add(id(value))
        _digest_value(value.__defaults__, owner, seen, out)
        _digest_value(value.__kwdefaults__, owner, seen, out)
        for cell in value.__closure__ or ():
            try:
                _digest_value(cell.cell_contents, owner, seen, out)
            except ValueError:  # empty cell
                out.append("<empty>")
    elif isinstance(value, type):
        out.append("T" + value.__module__ + "." + value.__qualname__)
        # Only classes defined by the owner hold state its import built.
        if value.__module__ == owner and id(value) not in seen:
            seen.add(id(value))
            for key in sorted(value.__dict__):
                if key in ("__dict__", "__weakref__", "__doc__", "__module__"):
                    continue
                out.append(key + "=")
                _digest_value(value.__dict__[key], owner, seen, out)
    elif isinstance(value, (types.ModuleType, types.BuiltinFunctionType, types.MethodDescriptorType)):
        out.append("M" + getattr(value, "__name__", ""))
    elif isinstance(value, types.MethodType):
        out.append("B" + value.__qualname__)
        _digest_value(value.__func__, owner, seen, out)
    elif isinstance(value, (types.MemberDescriptorType, types.GetSetDescriptorType)):
        out.append("D" + value.__name__)
    elif hasattr(value, "name") and hasattr(value, "value") and type(value).__module__.startswith("sqlglot"):
        out.append(repr(value))  # enum members
    elif type(value).__module__.startswith("sqlglot"):
        # Other sqlglot objects (dialect/tokenizer instances, tries, ...): compare their attributes.
        if id(value) in seen:
            out.append("<seen>")
            return
        seen.add(id(value))
        out.append("O" + type(value).__qualname__)
        attrs = dict(vars(value)) if hasattr(value, "__dict__") else {}
        for cls in type(value).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if hasattr(value, slot):
                    attrs[slot] = getattr(value, slot)
        _digest_value(attrs, owner, seen, out)
    elif isinstance(value, functools.partial):
        out.append("P[")
        _digest_value(value.func, owner, seen, out)
        _digest_value(value.args, owner, seen, out)
        _digest_value(value.keywords, owner, seen, out)
        out.append("]")
    elif isinstance(value, dataclasses.Field):
        out.append("DF" + str(value.name) + "|")
        if value.default is not dataclasses.MISSING:
            _digest_value(value.default, owner, seen, out)
        if value.default_factory is not dataclasses.MISSING:
            _digest_value(value.default_factory, owner, seen, out)
    elif type(value).__name__ == "_abc_data" or type(value).__name__ in ("RLock", "lock", "_RLock"):
        # ABC registry caches and import locks: runtime bookkeeping, not state the import built.
        out.append("<" + type(value).__name__ + ">")
    else:
        # Typing constructs, __future__ features and the like: a repr without an address is stable.
        text = repr(value)
        if " at 0x" in text:
            raise Undigestable(f"{owner}: {type(value).__module__}.{type(value).__qualname__}")
        out.append(text)


def module_digest(name: str) -> str | None:
    """Digest of a module's globals (and of the classes it defines), or None if not digestable."""
    module = sys.modules.get(name)
    if module is None:
        return None
    out: list[str] = []
    try:
        for key in sorted(vars(module)):
            if key.startswith("__"):
                continue
            out.append(key + "=")
            _digest_value(vars(module)[key], name, set(), out)
    except (Undigestable, RecursionError):
        return None
    return hashlib.blake2b("\n".join(out).encode(), digest_size=16).hexdigest()


def digests_for(names: list[str]) -> dict[str, str | None]:
    """Digest `names` (in sorted order). Dicts are visited in key order and set elements on
    their own `seen` copy, so a module's digest does not depend on what was digested before."""
    return {name: module_digest(name) for name in sorted(set(names))}


def import_all_sqlglot() -> None:
    """Import every sqlglot module, the way the coverage map does before any test."""
    import importlib
    import pkgutil
    import unittest  # noqa: F401  (tests run with unittest loaded: sqlglot reads that at import)

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name != "sqlglot.__main__":
            try:
                importlib.import_module(info.name)
            except Exception:
                pass


def import_minimal_sqlglot(mod_names: list[str]) -> None:
    """Import only `mod_names` (with unittest loaded, matching test runs)."""
    import importlib
    import unittest  # noqa: F401

    for name in mod_names:
        try:
            importlib.import_module(name)
        except Exception:
            pass


def _candidate_min_mods(owner: str) -> list[str]:
    mods: list[str] = []
    if owner.startswith(("sqlglot.generators.", "sqlglot.parsers.", "sqlglot.typing.")):
        stem = owner.rsplit(".", 1)[-1]
        if (SOURCE / "sqlglot" / "dialects" / f"{stem}.py").is_file():
            mods.append(f"sqlglot.dialects.{stem}")
    if owner not in mods:
        mods.append(owner)
    return mods


def _probe_min_batch(items: list[tuple[str, list[str]]]) -> dict[str, str | None]:
    """Fork a clean child per (owner, mods) before sqlglot is imported in this worker."""
    sys.path.insert(0, str(SOURCE))
    out: dict[str, str | None] = {}
    for owner, mods in items:
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r_fd)
            d = None
            try:
                import_minimal_sqlglot(mods)
                d = module_digest(owner)
            except BaseException:
                d = None
            finally:
                os.write(w_fd, marshal.dumps(d))
                os._exit(0)
        os.close(w_fd)
        chunks = []
        while chunk := os.read(r_fd, 4096):
            chunks.append(chunk)
        os.close(r_fd)
        os.waitpid(pid, 0)
        try:
            out[owner] = marshal.loads(b"".join(chunks))
        except Exception:
            out[owner] = None
    return out


def main() -> None:
    """Record line owners and owner digests at the pinned commit."""
    src_prefix = str(SOURCE.resolve()) + "/"
    owners: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    def tracer(frame, event, _arg):
        filename = frame.f_code.co_filename
        if not filename.startswith(src_prefix):
            return None
        rel = filename[len(src_prefix) :]
        if not rel.startswith("sqlglot/"):
            return None

        def local(frame, event, _arg):
            if event == "line":
                f = frame
                while f is not None and f.f_code.co_name != "<module>":
                    f = f.f_back
                if f is not None:
                    owners[rel][frame.f_lineno].add(f.f_globals.get("__name__", ""))
            return local

        return local

    sys.path.insert(0, str(SOURCE))
    sys.settrace(tracer)
    try:
        import_all_sqlglot()
    finally:
        sys.settrace(None)

    owner_names = sorted({o for per_line in owners.values() for names in per_line.values() for o in names})
    digests = {name: module_digest(name) for name in owner_names}  # canonical (sorted) order
    # Keep only digests that also come out the same in fresh processes with other hash seeds.
    for seed in ("1", "2"):
        other = marshal.loads(
            subprocess.run(
                [sys.executable, "-m", "blast_radius.import_state", "--digests", *owner_names],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                check=True,
            ).stdout
        )
        digests = {name: d for name, d in digests.items() if d is not None and other.get(name) == d}
    # And the same in the runtime checker itself, from a fresh process (no sqlglot imported yet).
    runtime = marshal.loads(
        subprocess.run(
            [sys.executable, "-m", "blast_radius.import_state", "--runtime", *owner_names],
            capture_output=True,
            check=True,
        ).stdout
    )
    digests = {name: d for name, d in digests.items() if runtime.get(name) == d}

    # Find owners whose minimal import closure produces the exact same deterministic digest (~15x faster runtime check).
    cand_items = [(o, _candidate_min_mods(o)) for o in sorted(digests)]
    min_matches = set(digests)
    for seed in ("1", "2"):
        got_min = marshal.loads(
            subprocess.run(
                [sys.executable, "-m", "blast_radius.import_state", "--min-worker"],
                input=marshal.dumps(cand_items),
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                check=True,
            ).stdout
        )
        min_matches = {o for o in min_matches if got_min.get(o) == digests[o]}
    min_imports = {o: mods for o, mods in cand_items if o in min_matches}

    data = {
        "owners": {rel: {ln: sorted(names) for ln, names in per.items()} for rel, per in owners.items()},
        "digests": digests,
        "min_imports": min_imports,
        "order": owner_names,
    }
    STATE_FILE.write_bytes(marshal.dumps(data))
    print(
        f"import state: {sum(len(v) for v in owners.values())} lines, {len(owner_names)} owners, "
        f"{len(data['digests'])} digestable, {len(min_imports)} minimal",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--digests"]:
        sys.path.insert(0, str(SOURCE))
        import_all_sqlglot()
        sys.stdout.buffer.write(marshal.dumps({name: module_digest(name) for name in sys.argv[2:]}))  # caller passes canonical order
    elif sys.argv[1:2] == ["--runtime"]:
        from blast_radius.verify import import_state_digests

        sys.stdout.buffer.write(marshal.dumps(import_state_digests({}, sys.argv[2:])))
    elif sys.argv[1:2] == ["--min-worker"]:
        items = marshal.loads(sys.stdin.buffer.read())
        sys.stdout.buffer.write(marshal.dumps(_probe_min_batch(items)))
    else:
        main()
