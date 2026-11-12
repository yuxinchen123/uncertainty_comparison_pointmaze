"""The report's contents table and its read / unread / updated marking.

Rule: ~/.claude/rules/auto-research-report-contents.md. A report is cut at its top-level headings (`## `). The
manifest `section_history.json`, beside the report, keeps per section: its content digest, when it first
appeared, when its text last changed, and the digest, text and figure digests it had when it was last marked
read. The states follow from content alone:

- unread: never marked read; the contents row and the whole section are blue in `analysis.html`;
- updated: marked read, but the text or a figure changed since; only the changed words are dark brown, and a
  changed figure gets a small blue "updated" label;
- read: marked read and unchanged.

`analysis.md` carries the contents table with the states in words; `analysis.html` carries the colours. Nothing
marks itself read: `python -m maze_campaign.analysis.contents --analysis-dir <dir> --mark-read all` (or section
identifiers) does, when the reader says so. It reads and writes text files only.

A report with another file name (a batch's `summary.md` and `summary.html`) passes that name without its ending as
`name` (`--name summary` on the command line); the manifest stays `section_history.json` in the report's folder.
"""

import argparse
from datetime import datetime
import difflib
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

HISTORY_FILE = 'section_history.json'
STYLE = '''body{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;line-height:1.5}
table{border-collapse:collapse;width:100%;margin:18px 0}
th,td{padding:6px;border-bottom:1px solid #bbb;text-align:left;vertical-align:top;overflow-wrap:anywhere}
img{max-width:100%;height:auto}section{margin-top:32px}
.unread,.unread a{color:#2452cc}.changed{color:#852D0F}.figure-updated{display:block;color:#2452cc}
@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}a{color:#adc7ff}
.unread,.unread a{color:#8bb4ff}.changed{color:#e7a48f}.figure-updated{color:#8bb4ff}}
@media print{body{max-width:none;margin:0}table{font-size:10pt}img{break-inside:avoid}}'''
MATH = ('<script>MathJax={tex:{inlineMath:[["$","$"]],displayMath:[["$$","$$"]]}};</script>'
        '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>')


def now():
    """The current time with its own offset, as stored in the manifest."""
    return datetime.now().astimezone().isoformat()


def pacific(stored):
    """A stored timestamp rendered for the reader: Pacific, with the zone named."""
    return datetime.fromisoformat(stored).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')


def section_identifier(title):
    """A section's link target, from its title."""
    # before: 'Rollout shape: environments per policy'; after: 'rollout-shape-environments-per-policy'.
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def split_sections(markdown):
    """Cut a report at its top-level headings: the text before the first, then (title, text) pairs."""
    # before: '# T\n\nintro\n\n## A\ntext a\n## B\ntext b\n'; after: ('# T\n\nintro\n\n', [('A', 'text a\n'), ('B', 'text b\n')]).
    parts = re.split(r'(?m)^## (.+)\n', markdown)
    return parts[0], list(zip(parts[1::2], parts[2::2]))


def figure_digests(analysis_dir, text):
    """A digest of every image file a section shows, by the path written in the section."""
    return {name: hashlib.sha256((Path(analysis_dir) / name).read_bytes()).hexdigest()
            for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)}


def status_of(record):
    """A section's state from its content: unread, read, or updated."""
    if not record.get('last_read_digest'):
        return 'unread'
    return 'read' if record['last_read_digest'] == record['content_digest'] else 'updated'


def update_history(history, preamble, sections, analysis_dir, timestamp):
    """Record every section's current text and figures; keep when it first appeared and what was last read."""
    identifiers = [section_identifier(title) for title, _ in sections]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f'Two sections of the report share a link target: {sorted(identifiers)}.')
    history['preamble'], history['order'] = preamble, identifiers
    for identifier, (title, text) in zip(identifiers, sections):
        figures = figure_digests(analysis_dir, text)
        digest = hashlib.sha256((text + json.dumps(figures, sort_keys=True)).encode()).hexdigest()
        saved = history['sections'].get(identifier, {})
        # The last-changed time moves only when the content digest does.
        changed = saved.get('content_digest') != digest
        history['sections'][identifier] = {
            **saved, 'title': title, 'content_digest': digest, 'current_text': text, 'current_figures': figures,
            'first_added': saved.get('first_added', timestamp),
            'last_changed': timestamp if changed else saved['last_changed']}
    return history


def contents_table(history, name='analysis'):
    """The contents table in Markdown: one row per section, in document order."""
    rows = ['## Contents\n\n| section | first written | last changed | status |\n|---|---|---|---|\n']
    for identifier in history['order']:
        record = history['sections'][identifier]
        rows.append(f"| [{record['title']}](#{identifier}) | {pacific(record['first_added'])} | "
                    f"{pacific(record['last_changed'])} | {status_of(record)} |\n")
    rows.append('\nTimes are Pacific, converted from machines running Eastern, and come from this report\'s own '
                'history of regenerations. [The same report with unread sections in blue and changed text in '
                f'brown]({name}.html). No section marks itself read.\n\n')
    return ''.join(rows)


class _VisibleText(HTMLParser):
    """Collect a page's visible text, without its tags."""

    def __init__(self):
        """Start with no text."""
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        """Keep the text in document order."""
        self.parts.append(data)


def visible_text(page):
    """The text a reader sees in a piece of HTML."""
    reader = _VisibleText()
    reader.feed(page)
    return ''.join(reader.parts)


def changed_ranges(previous, current):
    """Character ranges of the current visible text that were inserted or replaced since the previous text."""
    # Words are compared, never characters, and a formula between dollar signs is one word, so a mark never
    # splits a formula. before: 'W is 0.31 at $B$' -> 'W is 0.35 at $B$'; after: [(5, 9)].
    token = re.compile(r'\$[^$]*\$|\s+|[^\s$]+|\$')
    old, new = token.findall(previous), token.findall(current)
    offsets = [0]
    for word in new:
        offsets.append(offsets[-1] + len(word))
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [(offsets[start], offsets[end]) for action, _, _, start, end in matcher.get_opcodes()
            if action in ('insert', 'replace')]


class _ChangedWriter(HTMLParser):
    """Write a page again, with the changed ranges of its visible text inside coloured spans."""

    def __init__(self, ranges):
        """Start at the first visible character."""
        super().__init__(convert_charrefs=True)
        # The parser keeps its own `offset` (a column in the source), so the place in the visible text has another name.
        self.ranges, self.parts, self.seen = ranges, [], 0

    def handle_starttag(self, tag, attributes):
        """Keep an opening tag as it was written."""
        self.parts.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attributes):
        """Keep an image or another self-closing tag as it was written."""
        self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        """Keep a closing tag."""
        self.parts.append(f'</{tag}>')

    def handle_data(self, data):
        """Cut the text at the ends of the changed ranges and colour the pieces inside them."""
        end = self.seen + len(data)
        inside = [(max(start, self.seen), min(stop, end)) for start, stop in self.ranges
                  if start < end and stop > self.seen]
        cuts = sorted({self.seen, end, *[edge for pair in inside for edge in pair]})
        for start, stop in zip(cuts, cuts[1:]):
            piece = html.escape(data[start - self.seen:stop - self.seen])
            changed = any(left <= start < right for left, right in inside)
            self.parts.append(f'<span class="changed">{piece}</span>' if changed else piece)
        self.seen = end


def highlight_changes(current, previous):
    """The current HTML with the text that changed since the previous HTML in spans of class `changed`."""
    writer = _ChangedWriter(changed_ranges(visible_text(previous), visible_text(current)))
    writer.feed(current)
    return ''.join(writer.parts)


def render(analysis_dir, history, title, name='analysis'):
    """Write <name>.md with its contents table, and <name>.html with the colours of the three states."""
    from markdown_it import MarkdownIt
    analysis_dir = Path(analysis_dir)
    renderer = MarkdownIt('commonmark', {'html': True}).enable('table')
    contents = contents_table(history, name)
    markdown, pages = [history['preamble'], contents], []
    for identifier in history['order']:
        record = history['sections'][identifier]
        state = status_of(record)
        markdown.append(f"## {record['title']}\n{record['current_text']}")
        body = renderer.render(record['current_text'])
        if state == 'updated':
            body = highlight_changes(body, renderer.render(record.get('last_read_text') or ''))
            # A figure whose image file changed since the last read gets a small label under it.
            for image, digest in record['current_figures'].items():
                if digest != record.get('last_read_figures', {}).get(image):
                    body = re.sub(r'(<img[^>]*src="' + re.escape(image) + r'"[^>]*>)',
                                  r'\1<small class="figure-updated">updated</small>', body)
        pages.append(f'<section id="{identifier}" class="{state}"><h2>{html.escape(record["title"])}</h2>{body}</section>')
    (analysis_dir / f'{name}.md').write_text(''.join(markdown))

    # The contents rows of unread sections are blue as well.
    contents_page = renderer.render(contents)
    for identifier in history['order']:
        state = status_of(history['sections'][identifier])
        contents_page = contents_page.replace(f'href="#{identifier}"', f'class="{state}" href="#{identifier}"')
    (analysis_dir / f'{name}.html').write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f'<style>{STYLE}</style>{MATH}{renderer.render(history["preamble"])}{contents_page}{"".join(pages)}</html>\n')


def read_history(analysis_dir):
    """The manifest beside the report; an empty one before the first report."""
    path = Path(analysis_dir) / HISTORY_FILE
    return json.loads(path.read_text()) if path.exists() else {'sections': {}, 'order': [], 'preamble': ''}


def write_history(analysis_dir, history):
    """Replace the manifest."""
    (Path(analysis_dir) / HISTORY_FILE).write_text(json.dumps(history, indent=1) + '\n')


def write_report(analysis_dir, markdown, title, name='analysis'):
    """Write a regenerated report: update the manifest from its sections, then render both files."""
    preamble, sections = split_sections(markdown)
    history = update_history(read_history(analysis_dir), preamble, sections, analysis_dir, now())
    write_history(analysis_dir, history)
    render(analysis_dir, history, title, name)
    return {identifier: status_of(history['sections'][identifier]) for identifier in history['order']}


def mark_read(analysis_dir, identifiers, title='Report', name='analysis'):
    """Mark the named sections (or 'all') as read in their current state, and render the report again."""
    history = read_history(analysis_dir)
    chosen = history['order'] if 'all' in identifiers else list(identifiers)
    unknown = [identifier for identifier in chosen if identifier not in history['order']]
    if unknown:
        raise ValueError(f'No section {unknown} in this report; its sections are {history["order"]}.')
    for identifier in chosen:
        record = history['sections'][identifier]
        record.update(last_read_digest=record['content_digest'], last_read_text=record['current_text'],
                      last_read_figures=record['current_figures'], marked_read_at=now())
    write_history(analysis_dir, history)
    render(analysis_dir, history, title, name)
    return chosen


def main():
    """Mark sections of a generated report as read."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--mark-read', nargs='+', required=True, help="section identifiers, or 'all'")
    parser.add_argument('--name', default='analysis', help="the report's file name without its ending (summary for a batch)")
    arguments = parser.parse_args()
    # The page's title is the report's first heading.
    heading = re.match(r'# (.+)', read_history(arguments.analysis_dir)['preamble'])
    marked = mark_read(arguments.analysis_dir, arguments.mark_read, heading.group(1) if heading else 'Report',
                       arguments.name)
    print(f'marked as read: {", ".join(marked)}')


if __name__ == '__main__':
    main()
