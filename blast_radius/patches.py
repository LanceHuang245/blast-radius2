"""Unified-diff parsing, in-memory application, and change blocks."""

from __future__ import annotations

from blast_radius import Everything


class FilePatch:
    def __init__(self):
        self.old: str | None = None
        self.new: str | None = None
        # Each hunk is (old_start, old_count, new_start, body_lines).
        self.hunks: list[tuple[int, int, int, list[str]]] = []
        self.unusual = False  # binary, mode change, rename, ...
        self.binary = False


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("diff --git "):
            current = FilePatch()
            files.append(current)
            parts = line.split(" ")
            if len(parts) == 4:
                current.old, current.new = parts[2][2:], parts[3][2:]
            continue
        if (
            line.startswith("--- ")
            and i < len(lines)
            and lines[i].startswith("+++ ")
            and (current is None or current.hunks)
        ):
            # Plain unified diff (no `diff --git` header): a ---/+++ pair opens the next file.
            current = FilePatch()
            files.append(current)
        if current is None:
            continue
        if line.startswith("--- "):
            name = line[4:].split("\t")[0]
            current.old = None if name == "/dev/null" else name[2:]
        elif line.startswith("+++ "):
            name = line[4:].split("\t")[0]
            current.new = None if name == "/dev/null" else name[2:]
        elif line.startswith("@@ "):
            header = line.split("@@")[1].split()
            old_start, old_count = _range(header[0][1:])
            new_start, new_count = _range(header[1][1:])
            orig_old_count = old_count
            body: list[str] = []
            while (old_count > 0 or new_count > 0) and i < len(lines):
                hunk_line = lines[i]
                i += 1
                tag = hunk_line[:1]
                if tag == "\\":
                    continue
                if tag in (" ", ""):
                    old_count -= 1
                    new_count -= 1
                elif tag == "-":
                    old_count -= 1
                elif tag == "+":
                    new_count -= 1
                else:
                    raise Everything("malformed hunk")
                body.append(hunk_line if hunk_line else " ")
            current.hunks.append((old_start, orig_old_count, new_start, body))
        elif line.startswith(("Binary files", "GIT binary patch")):
            current.unusual = current.binary = True
        elif line.startswith((
            "old mode",
            "new mode",
            "rename from",
            "copy from",
            "similarity index",
        )):
            current.unusual = True
    return files


def _range(text: str) -> tuple[int, int]:
    if "," in text:
        start, count = text.split(",")
        return int(start), int(count)
    return int(text), 1


def apply(old_text: str, hunks: list[tuple[int, int, int, list[str]]]) -> str:
    """Apply parsed unified-diff hunks in memory, handling 0-context pure insertions."""
    old_lines = old_text.split("\n")
    out: list[str] = []
    cursor = 0
    for old_start, old_count, _new_start, body in hunks:
        if old_count == 0:
            start = old_start
        else:
            start = old_start - 1 if old_start > 0 else 0
        if start < cursor or start > len(old_lines):
            raise Everything("overlapping or out-of-range hunks")
        out.extend(old_lines[cursor:start])
        cursor = start
        for line in body:
            tag, content = line[0], line[1:]
            if tag in (" ", "-"):
                if cursor >= len(old_lines) or old_lines[cursor].rstrip("\r") != content.rstrip("\r"):
                    raise Everything("patch does not apply")
                cursor += 1
                if tag == " ":
                    out.append(content)
            else:
                out.append(content)
    out.extend(old_lines[cursor:])
    return "\n".join(out)


def change_blocks(hunks):
    """Runs of -/+ lines as ([(old_no, text)], [(new_no, anchor_old_no, text)]).

    anchor_old_no is the old line the insertion sits in front of.
    """
    for old_start, old_count, new_start, body in hunks:
        old_no = (old_start + 1) if old_count == 0 else (old_start if old_start > 0 else 1)
        new_no = new_start if new_start > 0 else 1
        removed: list[tuple[int, str]] = []
        added: list[tuple[int, int, str]] = []
        for line in body:
            tag, content = line[0], line[1:]
            if tag == " ":
                if removed or added:
                    yield removed, added
                    removed, added = [], []
                old_no += 1
                new_no += 1
            elif tag == "-":
                removed.append((old_no, content))
                old_no += 1
            else:
                added.append((new_no, old_no, content))
                new_no += 1
        if removed or added:
            yield removed, added
