"""Track when each section of the report was first written and last changed.

The report is regenerated from measurement files, so a plain file timestamp says nothing about
any individual section. This keeps a manifest instead: for every section, the content as it was
last seen, the time it first appeared, and the time its content last changed. The generator calls
`stamp` with the freshly built sections; anything whose text differs from the manifest gets a new
modification time, anything new gets both times set.

The manifest is seeded once from the report's git history (`seed_from_git`), so the recorded times
for sections written earlier in the project are the real commit times rather than the moment the
tracking was introduced.
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "section_times.json"
SNAPSHOT = HERE / "section_read_text.json"   # each section's text as the reader last read it
REPO = HERE.parents[3]   # .../09_parallelization/report/<run>/code -> the repository root
REL = "09_parallelization/report/2026-08-15-pointmaze-gpu-parallelization/report.md"


def split_sections(md: str) -> dict:
    """Split a report into its level-two sections, keyed by heading text.

    before: the whole document; after: {"2. The environment": "## 2. The environment\\n..."}
    Text before the first level-two heading is keyed "(title and introduction)".
    """
    out, key, buf = {}, "(title and introduction)", []
    for line in md.splitlines(keepends=True):
        if line.startswith("## "):
            out[key] = "".join(buf)
            key, buf = line[3:].strip(), [line]
        else:
            buf.append(line)
    out[key] = "".join(buf)
    return out


def _digest(text: str) -> str:
    """Content fingerprint of one section, ignoring trailing whitespace differences."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def load() -> dict:
    """The manifest, or an empty one."""
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def seed_from_git() -> dict:
    """Build the manifest from the report's git history.

    Walks every commit that touched the report, oldest first, splitting each version into
    sections. A section's first_added is the time of the first commit containing it; its
    last_modified is the time of the most recent commit in which its text changed.
    """
    commits = subprocess.run(
        ["git", "-C", str(REPO), "log", "--reverse", "--format=%H %cI", "--", REL],
        capture_output=True, text=True, check=True).stdout.split("\n")
    manifest = {}
    for line in commits:
        if not line.strip():
            continue
        sha, when = line.split(" ", 1)
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:{REL}"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            continue
        for name, text in split_sections(blob.stdout).items():
            d = _digest(text)
            if name not in manifest:
                manifest[name] = {"first_added": when, "last_modified": when, "digest": d}
            elif manifest[name]["digest"] != d:
                manifest[name].update(last_modified=when, digest=d)
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return manifest


def stamp(sections: dict, now: str = None) -> dict:
    """Update the manifest against freshly generated sections; return it.

    sections: {heading text: section markdown}. A heading whose text is unchanged keeps its
    recorded times; changed text updates last_modified; a new heading gets both times.
    """
    now = now or datetime.now().astimezone().isoformat(timespec="minutes")
    manifest = load()
    for name, text in sections.items():
        d = _digest(text)
        if name not in manifest:
            manifest[name] = {"first_added": now, "last_modified": now, "digest": d}
        elif manifest[name]["digest"] != d:
            manifest[name].update(last_modified=now, digest=d)
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return manifest


def anchor(heading: str) -> str:
    """The in-page link target a markdown renderer gives a heading."""
    slug = re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
    return re.sub(r"-+", "-", slug)


def short(iso: str, seconds: bool = False) -> str:
    """A stored timestamp rendered for display: Pacific Time with an explicit marker.

    The servers run on Eastern Time, so every stored timestamp carries an Eastern offset;
    the reader works in Pacific. The stored value keeps its own zone (nothing is lost); only
    this display converts.

    before: "2026-08-15T13:39:02-04:00"  after: "2026-08-15 10:39 PT"
    """
    if not iso:
        return "—"
    when = datetime.fromisoformat(iso).astimezone(ZoneInfo("America/Los_Angeles"))
    return when.strftime("%Y-%m-%d %H:%M:%S PT" if seconds else "%Y-%m-%d %H:%M PT")


def figure_digests() -> dict:
    """A fingerprint of every figure file, so a redrawn plot can be told from an unchanged one.

    The markdown that places a figure never changes when the figure is redrawn, so the text
    comparison that marks prose cannot see it. The image bytes can.
    before: figures/cpu_vs_gpu.png on disk; after: {"cpu_vs_gpu.png": "3f9a1c4d..."}
    """
    figs = HERE.parent / "figures"
    if not figs.exists():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(figs.glob("*.png"))}


def changed_figures() -> set:
    """Names of the figures whose image differs from the version last marked read.

    A snapshot taken before figures were tracked holds no record of them at all. That is not
    evidence every figure changed — it is no evidence either way — so nothing is marked until
    the reader next marks the document read and a baseline exists.
    """
    was = read_text().get("__figures__")
    if not was:
        return set()
    return {name for name, d in figure_digests().items() if was.get(name) != d}


def read_text() -> dict:
    """Each section's text as it stood when the reader last marked it read."""
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}


def state(name: str, manifest: dict) -> str:
    """Whether the reader has seen this section in its current form.

    Computed from content, never from a flag someone has to remember to set:
    before: manifest["X"] = {"digest": "abc", "read_digest": "abc"}   after: "read"
    before: manifest["X"] = {"digest": "def", "read_digest": "abc"}   after: "updated"
    before: manifest["X"] = {"digest": "def"}  (never marked read)    after: "unread"
    """
    t = manifest.get(name, {})
    if not t.get("read_digest"):
        return "unread"
    return "read" if t["read_digest"] == t["digest"] else "updated"


def mark_all_read(sections: dict, now: str = None) -> dict:
    """Record that the reader has read every section in its current form.

    Stores both the digest (for the cheap state check) and the text itself (so the next
    regeneration can show WHICH parts changed, not merely that something did).
    """
    now = now or datetime.now().astimezone().isoformat(timespec="minutes")
    manifest = load()
    for name, text in sections.items():
        entry = manifest.setdefault(name, {"first_added": now, "last_modified": now,
                                           "digest": _digest(text)})
        entry.update(read_digest=_digest(text), read_at=now)
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    # the figures go under a reserved key beside the sections: they are part of what the reader
    # just read, and their images are the only way to tell later that a plot was redrawn
    SNAPSHOT.write_text(json.dumps(dict(sections, __figures__=figure_digests()), indent=1))
    return manifest


STATE_LABEL = {"read": "read", "updated": "updated", "unread": "unread"}


def table_of_contents(order, manifest) -> str:
    """The contents table: one row per section, with its times and whether it has been read.

    An unread section's title is written in blue and an updated one's in dark brown, through
    span classes the rendered page styles; in plain markdown the class names are inert and the
    status column still carries the same information.
    """
    md = ["## Contents\n",
          "| section | first written | last changed | status |", "|---|---|---|---|"]
    for name in order:
        if name == "(title and introduction)":
            continue
        t = manifest.get(name, {})
        st = state(name, manifest)
        link = f"[{name}](#{anchor(name)})"
        title = link if st == "read" else f'<span class="{st}">{link}</span>'
        md.append(f"| {title} | {short(t.get('first_added'))} "
                  f"| {short(t.get('last_modified'))} | {STATE_LABEL[st]} |")
    md.append("\n*Times are when a section's text first appeared in this document and when it "
              "last changed, taken from the document's version history. A section whose numbers "
              "were re-measured shows a later change time. All times are Pacific (PT); the "
              "machines that produced them run on Eastern Time and the values are converted "
              "for display.*")
    md.append("\n*Status is computed by comparing each section against the text last marked read: "
              "an **unread** section is written in blue, title and body; in an **updated** section "
              "the text that changed since you read it is written in dark brown, and the rest is "
              "left alone.*\n")
    return "\n".join(md)


def blocks(text: str) -> list:
    """Split a section into renderable blocks: paragraphs, tables, headings, images, code.

    Blank lines separate blocks, except inside a fenced code block, where a blank line is part
    of the code. Splitting this way lets the page colour only the blocks that changed, and keeps
    a markdown table whole — a table split across two spans stops being a table.
    before: "para one\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    after:  ["para one", "| a | b |\n|---|---|\n| 1 | 2 |"]
    """
    out, buf, fenced = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not line.strip() and not fenced:
            if buf:
                out.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def changed_blocks(current: str, previously_read: str) -> set:
    """Indices of the blocks of `current` that are new or altered since the read version.

    before: read "A\n\nB", current "A\n\nB2\n\nC"   after: {1, 2}
    """
    # an opcode carries a range in each sequence; the ones that matter here are j1..j2, the
    # positions in the CURRENT text, since those are the blocks about to be rendered
    now, before = blocks(current), blocks(previously_read)
    changed = set()
    for tag, _, _, j1, j2 in SequenceMatcher(None, before, now, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))
    return changed


if __name__ == "__main__":
    import sys
    if "--mark-read" in sys.argv:
        # the reader says they have read the document as it now stands: record every section's
        # current text, so the next regeneration can colour only what changes from here on
        # the contents table is excluded because it is regenerated from this very state and
        # would always disagree with itself; the title and introduction ARE tracked, since the
        # reader reads them like any other part
        secs = {k: v for k, v in
                split_sections((HERE.parent / "report.md").read_text()).items()
                if k != "Contents"}
        man = mark_all_read(secs)
        print(f"marked {len(secs)} sections read at {short(man[next(iter(secs))]['read_at'])}")
    else:
        m = seed_from_git()
        for k, v in m.items():
            print(f"{short(v['first_added'])}  {short(v['last_modified'])}  {k}")
