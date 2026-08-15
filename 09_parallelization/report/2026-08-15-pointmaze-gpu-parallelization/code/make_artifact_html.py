"""Render report.md into a self-contained HTML page (figures embedded as data URIs).

Design: utilitarian technical-report treatment. Single accent = the report's own chart
blue; serif body for reading, sans headings, monospace tabular numerals in tables; light
and dark themes token-driven (bare :root = light; media query + [data-theme] overrides).

Run: <python with markdown> make_artifact_html.py  ->  ../report_page.html
"""
import base64
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent

md_text = (REPORT / "report.md").read_text()
body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

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
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #16181c; --panel: #1e2126; --ink: #e8e8e4; --ink-2: #9aa0ab;
    --rule: #2a2d33; --accent: #5598e7; --accent-ink: #86b6ef;
  }
}
:root[data-theme="dark"] {
  --ground: #16181c; --panel: #1e2126; --ink: #e8e8e4; --ink-2: #9aa0ab;
  --rule: #2a2d33; --accent: #5598e7; --accent-ink: #86b6ef;
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

meta = ('<div class="meta">'
        '<span>hardware <b>H100 NVL, serval05</b></span>'
        '<span>date <b>2026-08-15</b></span>'
        '<span>repo <b>RND/09_parallelization</b></span>'
        '<span>source <b>report.md (generated from benchmark JSONs)</b></span>'
        '</div>')
body = body.replace("</h1>", "</h1>" + meta, 1)

html = f"<title>PointMaze GPU Parallelization</title>\n<style>{CSS}</style>\n<main>{body}</main>\n"
(REPORT / "report_page.html").write_text(html)
print("wrote", REPORT / "report_page.html", len(html), "bytes")
