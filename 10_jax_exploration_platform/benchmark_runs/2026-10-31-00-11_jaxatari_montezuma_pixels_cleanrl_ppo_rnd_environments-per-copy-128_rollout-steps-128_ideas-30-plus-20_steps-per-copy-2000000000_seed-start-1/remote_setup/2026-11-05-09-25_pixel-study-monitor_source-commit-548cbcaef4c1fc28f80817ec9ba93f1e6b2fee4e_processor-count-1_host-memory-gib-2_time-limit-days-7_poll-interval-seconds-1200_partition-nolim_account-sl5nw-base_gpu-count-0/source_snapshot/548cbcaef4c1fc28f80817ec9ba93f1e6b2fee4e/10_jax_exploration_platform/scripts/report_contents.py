"""Keep the development document's contents table and its read / unread / updated marking current.

Rule: ~/.claude/rules/auto-research-report-contents.md. One row per subsection: its title (a link),
when its text first appeared and last changed (from the document's git history, shown in Pacific
time), and its status:

- unread: never marked read; the row's title and the whole subsection render blue;
- updated: marked read, but its text changed since; only the changed blocks render dark brown;
- read: marked read and unchanged.

A subsection's text runs from its \\subsection line to the next \\subsection, \\section or
\\end{document}; blocks are separated by blank lines. Marking lines this script inserts end in
"% REPORT-STATUS" and are removed before any text is compared, so the marking never counts as a
change. Nothing marks itself read: `--mark-read <label|all>` does, when the reader says so.

Usage: report_contents.py <document.tex> [--mark-read LABEL ...]
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TAG = "% REPORT-STATUS"
START = "% >>> REPORT CONTENTS START"
END = "% <<< REPORT CONTENTS END"
HEADING = re.compile(r"^\\(sub)?section\{(.*)\}\s*$")


def strip_marks(text: str) -> str:
    """The document without this script's marking lines and without its contents table."""
    text = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)
    return "\n".join(line for line in text.split("\n") if not line.rstrip().endswith(TAG))


def subsections(text: str) -> list:
    """[(label, title, body)] for every subsection of the (unmarked) document, in order.

    before: "\\subsection{Training speed}\\n\\label{sec:x}\\nText.\\n\\n\\subsection{...}"
    after:  [("sec:x", "Training speed", "\\label{sec:x}\\nText.\\n"), ...]
    """
    lines = text.split("\n")
    found, current = [], None
    for line in lines:
        m = HEADING.match(line)
        if m or line.startswith(r"\end{document}"):
            if current:
                found.append(current)
            current = [m.group(2), []] if (m and m.group(1)) else None
            continue
        if current is not None:
            current[1].append(line)
    out = []
    for title, body in found:
        label = re.search(r"\\label\{(sec:[^}]*)\}", "\n".join(body))
        if label:
            out.append((label.group(1), title, "\n".join(body)))
    return out


def digest(body: str) -> str:
    """A content fingerprint that ignores trailing spaces and blank-line runs."""
    norm = "\n".join(l.rstrip() for l in body.strip().split("\n"))
    return hashlib.sha256(re.sub(r"\n{3,}", "\n\n", norm).encode()).hexdigest()[:16]


def history(path: Path) -> list:
    """[(commit time ISO, {label: digest})] oldest first, then the working copy if it differs."""
    repo = Path(subprocess.check_output(["git", "-C", str(path.parent), "rev-parse",
                                         "--show-toplevel"], text=True).strip())
    rel = path.resolve().relative_to(repo)
    log = subprocess.check_output(["git", "-C", str(repo), "log", "--follow", "--name-only",
                                   "--format=@%H %cI", "--", str(rel)], text=True)
    versions, commit = [], None
    for line in log.split("\n"):
        if line.startswith("@"):
            commit = line[1:].split()
        elif line.strip() and commit:
            text = subprocess.run(["git", "-C", str(repo), "show", f"{commit[0]}:{line.strip()}"],
                                  capture_output=True, text=True).stdout
            versions.append((commit[1], {l: digest(b) for l, _, b in subsections(strip_marks(text))}))
            commit = None
    versions.reverse()
    current = {l: digest(b) for l, _, b in subsections(strip_marks(path.read_text()))}
    if not versions or versions[-1][1] != current:
        versions.append((datetime.now().astimezone().isoformat(), current))
    return versions


def pacific(iso: str) -> str:
    """A stored timestamp rendered for the reader: Pacific, with the zone named."""
    return datetime.fromisoformat(iso).astimezone(
        ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")


def blocks(body: str) -> list:
    """The body cut at blank lines, each block with trailing spaces removed."""
    return [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]


def color_floats(text: str) -> str:
    """Put a color line right after every table or figure opening, since floats drop the text color.

    before: "\\begin{table}[H]\n\\centering"; after: "\\begin{table}[H]\n\\studyfloatcolor % REPORT-STATUS\n\\centering"
    """
    return re.sub(r"^(\\begin\{(?:table|figure)\}.*)$", lambda m: m.group(1) + f"\n\\studyfloatcolor {TAG}",
                  text, flags=re.M)


def render(path: Path, state: dict) -> None:
    """Rewrite the contents table and every marking from the current text and the read state."""
    raw = strip_marks(path.read_text())
    secs = subsections(raw)
    versions = history(path)
    rows, status_of = [], {}
    for label, title, body in secs:
        seen = [(t, d[label]) for t, d in versions if label in d]
        first = seen[0][0]
        last = next((seen[k][0] for k in range(len(seen) - 1, 0, -1)
                     if seen[k][1] != seen[k - 1][1]), first)
        read = state.get(label)
        status = ("unread" if read is None else
                  "read" if read["digest"] == digest(body) else "updated")
        status_of[label] = (status, read)
        name = rf"\hyperref[{label}]{{{title}}}"
        if status == "unread":
            name = rf"\textcolor{{unreadblue}}{{{name}}}"
        rows.append(f"{name} & {pacific(first)} & {pacific(last)} & {status} \\\\")

    # the contents table, placed after the table of contents
    # (a plain block, not a float: a numbered table here would sit outside every section)
    table = "\n".join([
        START,
        r"\begin{center}", r"\small",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{6.4cm} l l l@{}}", r"\toprule",
        r"\textbf{subsection} & \textbf{first written} & \textbf{last changed} & \textbf{status} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{center}",
        r"{\small Times come from the document's version history and are Pacific, converted "
        r"from machines running Eastern. Unread subsections are \textcolor{unreadblue}{blue}; "
        r"in updated ones only the changed text is \textcolor{editbrown}{brown}.\par}",
        END])
    out = raw.replace("\\tableofcontents\n", "\\tableofcontents\n" + table + "\n", 1)

    # the markings: a whole unread subsection, or the changed blocks of an updated one
    for label, title, body in secs:
        status, read = status_of[label]
        if status == "read":
            continue
        heading = f"\\subsection{{{title}}}\n"
        if status == "unread":
            marked = (f"\\studyunreadbegin {TAG}\n" + color_floats(body.rstrip("\n"))
                      + f"\n\\studyunreadend {TAG}\n")
        else:
            old = set(blocks(read["text"]))
            parts = []
            for block in re.split(r"(\n\s*\n)", body):
                if block.strip() and block.strip() not in old:
                    parts.append(f"\\studychangedbegin {TAG}\n{color_floats(block)}\n"
                                 f"\\studychangedend {TAG}")
                else:
                    parts.append(block)
            marked = "".join(parts)
        out = out.replace(heading + body, heading + marked, 1)
    path.write_text(out)


def main():
    """Mark sections read when asked, then rewrite the contents and the markings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("--mark-read", nargs="*", default=[])
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args()
    state_path = args.state or args.document.with_name("report_read_state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.mark_read:
        secs = subsections(strip_marks(args.document.read_text()))
        for label, _, body in secs:
            if label in args.mark_read or "all" in args.mark_read:
                state[label] = {"digest": digest(body), "text": body,
                                "marked_at": datetime.now().astimezone().isoformat()}
        state_path.write_text(json.dumps(state, indent=1) + "\n")
    render(args.document, state)


if __name__ == "__main__":
    main()
