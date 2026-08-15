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
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "section_times.json"
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


def short(iso: str) -> str:
    """An ISO timestamp shortened for a table cell: 2026-08-15 04:45."""
    return iso[:16].replace("T", " ") if iso else "—"


def table_of_contents(order, manifest) -> str:
    """The contents table: one row per section, with when it appeared and when it last changed."""
    md = ["## Contents\n",
          "| section | first written | last changed |", "|---|---|---|"]
    for name in order:
        if name == "(title and introduction)":
            continue
        t = manifest.get(name, {})
        md.append(f"| [{name}](#{anchor(name)}) | {short(t.get('first_added'))} "
                  f"| {short(t.get('last_modified'))} |")
    md.append("\n*Times are when a section's text first appeared in this document and when it "
              "last changed, taken from the document's version history. A section whose numbers "
              "were re-measured shows a later change time.*\n")
    return "\n".join(md)


if __name__ == "__main__":
    m = seed_from_git()
    for k, v in m.items():
        print(f"{short(v['first_added'])}  {short(v['last_modified'])}  {k}")
