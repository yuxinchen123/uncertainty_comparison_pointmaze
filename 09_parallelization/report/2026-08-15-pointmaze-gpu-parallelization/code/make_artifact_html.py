"""Render report.md into a self-contained HTML page (figures embedded as data URIs).

Design: utilitarian technical-report treatment. Single accent = the report's own chart
blue; serif body for reading, sans headings, monospace tabular numerals in tables; light
and dark themes token-driven (bare :root = light; media query + [data-theme] overrides).

Run: <python with markdown> make_artifact_html.py  ->  ../report_page.html
"""
import base64
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent

import section_times as st

md_text = (REPORT / "report.md").read_text()


def render(md: str) -> str:
    """One markdown fragment to HTML, with the extensions this document needs."""
    return markdown.markdown(md, extensions=["tables", "fenced_code"])


def render_document(md: str) -> str:
    """The document as HTML, with unread sections in blue and changed text in dark brown.

    Each section is converted on its own so that a colour can be applied to part of a document
    without disturbing the rest, and each block within a changed section is converted on its own
    so the colour lands on the blocks that actually changed. Converting block by block (rather
    than wrapping markdown in a div) is what keeps tables working: python-markdown does not look
    for markdown inside a raw HTML block, so a table wrapped in a div would render as text.
    before: section "X" is updated, its second paragraph changed
    after:  ...<div class="updated"><p>that paragraph</p></div>... and the rest plain
    """
    manifest, snapshot = st.load(), st.read_text()
    out = []
    for name, text in st.split_sections(md).items():
        # the contents table carries the state of everything else and has none of its own
        status = "read" if name == "Contents" or name not in manifest else st.state(name, manifest)
        if status == "read":
            out.append(render(text))
        elif status == "unread":
            # nothing here has been seen before, so the whole section carries the colour
            out.append(f'<div class="unread">{render(text)}</div>')
        else:
            changed = st.changed_blocks(text, snapshot.get(name, ""))
            pieces = []
            for i, block in enumerate(st.blocks(text)):
                html_block = render(block)
                pieces.append(f'<div class="updated">{html_block}</div>' if i in changed
                              else html_block)
            out.append("\n".join(pieces))
    return "\n".join(out)


body = render_document(md_text)

# embed each figure as a data URI so the page needs no filesystem
def embed(m):
    p = REPORT / m.group(2)
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img alt="{m.group(1)}" src="data:image/png;base64,{data}"'

body = re.sub(r'<img alt="([^"]*)" src="(figures/[^"]+)"', embed, body)

CSS = """
:root {
  --ground: #fbfaf8; --panel: #f1efe9; --ink: #1c1e21; --ink-2: #5c6270;
  --rule: #e3e1dc; --accent: #2a78d6; --accent-ink: #1c5cab;
  --unread: #1544c4; --updated: #852d0f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #16181c; --panel: #1e2126; --ink: #e8e8e4; --ink-2: #9aa0ab;
    --rule: #2a2d33; --accent: #5598e7; --accent-ink: #86b6ef;
    /* both colours are lightened for a dark ground; each stays recognisably the same colour */
    --unread: #7ea8f6; --updated: #d98b62;
  }
}
:root[data-theme="dark"] {
  --ground: #16181c; --panel: #1e2126; --ink: #e8e8e4; --ink-2: #9aa0ab;
  --rule: #2a2d33; --accent: #5598e7; --accent-ink: #86b6ef;
  --unread: #7ea8f6; --updated: #d98b62;
}
* { box-sizing: border-box; }
body {
  background: var(--ground); color: var(--ink); margin: 0;
  font-family: Charter, Georgia, 'Times New Roman', serif;
  font-size: 16.5px; line-height: 1.55;
}
main { max-width: 880px; margin: 0 auto; padding: 3rem 1.5rem 5rem; }
h1, h2, h3 {
  font-family: 'Helvetica Neue', Helvetica, Arial, system-ui, sans-serif;
  line-height: 1.2; text-wrap: balance; letter-spacing: -0.01em;
}
h1 { font-size: 1.9rem; font-weight: 650; margin: 0 0 1rem; }
h2 {
  font-size: 1.25rem; font-weight: 650; margin: 3rem 0 0.75rem;
  padding-top: 1.25rem; border-top: 1px solid var(--rule);
}
p, li { max-width: 72ch; }
a { color: var(--accent-ink); }
code {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.86em; background: var(--panel); padding: 0.08em 0.32em;
  border-radius: 3px;
}
.tablewrap { overflow-x: auto; margin: 1.1rem 0; }
table {
  border-collapse: collapse; font-size: 0.83rem; line-height: 1.4;
  font-family: 'Helvetica Neue', Helvetica, Arial, system-ui, sans-serif;
}
th, td {
  padding: 0.42rem 0.8rem; text-align: left; vertical-align: top;
  border-bottom: 1px solid var(--rule); max-width: 34rem;
}
td { font-variant-numeric: tabular-nums; }
th {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ink-2); font-weight: 600; border-bottom: 2px solid var(--accent);
}
img { max-width: 100%; height: auto; display: block; margin: 1.25rem 0; }
/* reading state: blue = not yet read, brown = changed since you read it. The brown is the
   colour the claude-edit skill uses for a machine-made replacement, RGB(133,45,15). */
.unread, .unread p, .unread li, .unread td, .unread th, .unread h2, .unread h3,
.unread em, .unread strong, .unread a { color: var(--unread); }
.updated, .updated p, .updated li, .updated td, .updated th, .updated h2, .updated h3,
.updated em, .updated strong, .updated a { color: var(--updated); }
.updated table th { border-bottom-color: var(--updated); }
ol, ul { padding-left: 1.4rem; }
li { margin: 0.3rem 0; }
em { color: var(--ink-2); }
.meta {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.78rem; color: var(--ink-2); margin-bottom: 2.25rem;
  display: flex; flex-wrap: wrap; gap: 0.5rem 1.75rem;
}
.meta span b { color: var(--ink); font-weight: 600; }
@media print { body { font-size: 12px; } }
"""

# wrap every table for horizontal scroll containment
body = body.replace("<table>", '<div class="tablewrap"><table>').replace(
    "</table>", "</table></div>")

# the header carries the time this page was built, in the reader's zone (the machines run
# on Eastern Time, so the value is converted rather than printed as the server sees it)
built = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
meta = ('<div class="meta">'
        '<span>hardware <b>H100 NVL, serval05</b></span>'
        f'<span>generated <b>{built}</b></span>'
        '<span>repo <b>RND/09_parallelization</b></span>'
        '<span>source <b>report.md (generated from benchmark JSONs)</b></span>'
        '</div>')
body = body.replace("</h1>", "</h1>" + meta, 1)

html = f"<title>PointMaze GPU Parallelization</title>\n<style>{CSS}</style>\n<main>{body}</main>\n"
(REPORT / "report_page.html").write_text(html)
print("wrote", REPORT / "report_page.html", len(html), "bytes")
