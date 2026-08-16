"""Render the survey as one self-contained HTML page, from the same measurements as results.md.

The page is generated, never hand-edited: it reads `data/throughput/` and `data/probe/` exactly
as `analyze.py` does, and embeds the figures as data URIs so it needs no network.
"""
import base64
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze import (RUN, PLOTS, PROBES, best_cpu_count, cell_of, cpu_counts_for,
                     load_classes, load_jobs)
from node_classes import COPY_COUNTS

PACIFIC = ZoneInfo("America/Los_Angeles")

# An instrument readout, not an essay: a cool teal-biased neutral set for the page, one deep
# teal that means "measured", one rust that means "this card could not hold the run". Monospace
# carries the headings because every proper noun on the page is a machine name or a timing.
STYLE = """
:root {
  --ground: #F2F6F6; --surface: #FFFFFF; --sunk: #E7EEEE;
  --ink: #132123; --muted: #5A7276; --rule: #D6E2E2;
  --accent: #0E757E; --accent-soft: #DCEEEF; --warn: #9C4520; --warn-soft: #F6E4DA;
  --plate: #FFFFFF;
  --shadow: 0 1px 2px rgba(19,33,35,.06), 0 8px 24px -16px rgba(19,33,35,.35);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0C1618; --surface: #132123; --sunk: #0F1B1D;
    --ink: #E2ECEC; --muted: #8CA5A8; --rule: #24383B;
    --accent: #4BC3CC; --accent-soft: #123336; --warn: #E0895C; --warn-soft: #33211A;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"] {
  --ground: #0C1618; --surface: #132123; --sunk: #0F1B1D;
  --ink: #E2ECEC; --muted: #8CA5A8; --rule: #24383B;
  --accent: #4BC3CC; --accent-soft: #123336; --warn: #E0895C; --warn-soft: #33211A;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 16px; line-height: 1.6; -webkit-font-smoothing: antialiased;
}
.mono, th, td.num, .tag, .eyebrow, h1, h2, .readout-value {
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px 96px; }
header.top { border-bottom: 1px solid var(--rule); background: var(--surface); }
header.top .wrap { padding-top: 56px; padding-bottom: 40px; }
.eyebrow {
  font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 18px;
}
h1 { font-size: clamp(28px, 3.6vw, 42px); line-height: 1.15; margin: 0 0 18px;
     font-weight: 600; letter-spacing: -.02em; text-wrap: balance; max-width: 20ch; }
.standfirst { font-size: 18px; color: var(--muted); margin: 0; max-width: 62ch; }
.readouts { display: grid; gap: 1px; background: var(--rule);
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            border: 1px solid var(--rule); border-radius: 10px; overflow: hidden;
            margin: 40px 0 0; }
.readout { background: var(--surface); padding: 18px 20px; }
.readout-label { font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
                 color: var(--muted); margin: 0 0 8px; }
.readout-value { font-size: 26px; font-weight: 600; letter-spacing: -.02em;
                 font-variant-numeric: tabular-nums; }
.readout-note { font-size: 13px; color: var(--muted); margin-top: 4px; }
section { margin-top: 56px; }
h2 { font-size: 13px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
     font-weight: 600; margin: 0 0 6px; display: flex; align-items: baseline; gap: 12px; }
h2 .n { color: var(--accent); }
h3 { font-size: 22px; margin: 0 0 14px; font-weight: 600; letter-spacing: -.01em;
     text-wrap: balance; }
p { max-width: 68ch; }
p.lede { font-size: 17px; }
.scroller { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px;
            background: var(--surface); box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th { text-align: left; font-weight: 600; font-size: 11px; letter-spacing: .07em;
     text-transform: uppercase; color: var(--muted); padding: 12px 14px;
     border-bottom: 1px solid var(--rule); white-space: nowrap; background: var(--sunk); }
td { padding: 10px 14px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.name { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
.best { font-weight: 700; color: var(--accent); }
.second { text-decoration: underline; text-underline-offset: 3px; }
.tag { display: inline-block; font-size: 11px; letter-spacing: .04em; padding: 2px 8px;
       border-radius: 999px; background: var(--accent-soft); color: var(--accent); }
.tag.warn { background: var(--warn-soft); color: var(--warn); }
figure { margin: 24px 0 0; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--rule);
             border-radius: 10px; background: var(--plate); }
figcaption { font-size: 13px; color: var(--muted); margin-top: 10px; }
.finding { border-left: 3px solid var(--accent); padding: 2px 0 2px 18px; margin: 26px 0;
           background: none; }
.finding strong { color: var(--accent); }
ul.plain { padding-left: 20px; }
ul.plain li { margin-bottom: 6px; max-width: 68ch; }
footer { margin-top: 72px; padding-top: 24px; border-top: 1px solid var(--rule);
         font-size: 13px; color: var(--muted); }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def data_uri(path):
    """A PNG embedded in the page, so the artifact needs no network."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(text):
    """Text placed into markup."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def table(headers, rows, numeric_from=1):
    """One table inside its own horizontal scroller, digits right-aligned and tabular."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            klass = "name" if i == 0 else ("num" if i >= numeric_from else "")
            cells.append(f'<td class="{klass}">{cell}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="scroller"><table><thead><tr>' + head + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def mark(value, ranked, fmt):
    """Best value in the accent colour, second best underlined — the report's own convention."""
    text = fmt.format(value)
    if ranked and value == ranked[0]:
        return f'<span class="best">{text}</span>'
    if len(ranked) > 1 and value == ranked[1]:
        return f'<span class="second">{text}</span>'
    return text


def memory_split(jobs, probes):
    """Card memory per thousand copies, split at compute capability 8.0 — the same numbers the
    markdown report computes, so the two never drift apart."""
    capability = {p["device_kind"]: float(p["compute_capability"]) for p in probes
                  if p.get("compute_capability")}
    per_thousand = {"older": [], "newer": []}
    largest = {"older": [], "newer": []}
    for job in jobs.values():
        for cell in job["cells"]:
            if cell.get("status") != "measured" or cell["device_kind"] not in capability:
                continue
            group = "newer" if capability[cell["device_kind"]] >= 8.0 else "older"
            gb = cell["peak_device_memory_mb"] / 1000
            per_thousand[group].append(gb / cell["n_copies"] * 1000)
            if cell["n_copies"] == max(COPY_COUNTS):
                largest[group].append(gb)
    out = {"older": float(np.mean(per_thousand["older"])),
           "newer": float(np.mean(per_thousand["newer"]))}
    out["ratio"] = out["older"] / out["newer"]
    if largest["older"] and largest["newer"]:
        out["largest_older"] = float(np.mean(largest["older"]))
        out["largest_newer"] = float(np.mean(largest["newer"]))
    return out


def build(classes, jobs, probes):
    """The whole page."""
    mem = memory_split(jobs, probes)
    now = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")
    n_cells = sum(1 for j in jobs.values() for c in j["cells"] if c.get("status") == "measured")
    n_oom = sum(1 for j in jobs.values() for c in j["cells"]
                if c.get("status", "").startswith(("out_of_memory", "not_attempted_smaller")))
    spreads = np.asarray([c["relative_spread_middle_half"] for j in jobs.values()
                          for c in j["cells"] if c.get("status") == "measured"])
    total_jobs = sum(len(cpu_counts_for(c)) for c in classes)

    # the headline readouts: the fastest card at the largest copy count everyone could reach
    top = None
    for cls in classes:
        cell, _ = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), 4096)
        if cell and (top is None or cell["total_steps_per_second"] > top[1]["total_steps_per_second"]):
            top = (cls, cell)

    out = [f"<title>PointMaze Trainer on 27 Graphics Cards</title>",
           f"<style>{STYLE}</style>",
           '<header class="top"><div class="wrap">',
           '<p class="eyebrow">JAX PPO+RND &middot; single update per rollout &middot; '
           'PointMaze Large</p>',
           "<h1>How fast the trainer runs on every card this cluster has</h1>",
           '<p class="standfirst">Every distinct graphics-card configuration of the '
           '<span class="mono">gpu</span> and <span class="mono">gnolim</span> partitions, '
           "measured at 512 to 4,096 training copies with 8, 16 and 32 processors. One node "
           "per configuration, because two nodes of one class are the same machine twice.</p>",
           '<div class="readouts">']

    if top:
        out.append(
            f'<div class="readout"><p class="readout-label">Fastest at 4,096 copies</p>'
            f'<div class="readout-value">{top[1]["total_steps_per_second"] / 1e6:.1f}M</div>'
            f'<p class="readout-note">environment steps per second &mdash; '
            f'{esc(top[0]["display_name"])}</p></div>')
        out.append(
            f'<div class="readout"><p class="readout-label">Ten million steps per copy</p>'
            f'<div class="readout-value">{10 * top[1]["hours_per_million_steps_per_copy"]:.2f} h'
            f'</div><p class="readout-note">on that card, 4,096 copies at once</p></div>')
    out += [
        f'<div class="readout"><p class="readout-label">Measurements</p>'
        f'<div class="readout-value">{n_cells}</div>'
        f'<p class="readout-note">{len(jobs)} of {total_jobs} jobs reported; {n_oom} cells did '
        f'not fit on their card</p></div>',
        f'<div class="readout"><p class="readout-label">Worst timing spread</p>'
        f'<div class="readout-value">{spreads.max() * 100:.2f}%</div>'
        f'<p class="readout-note">middle half of rounds, over all {len(spreads)} cells; '
        f'{np.median(spreads) * 100:.2f}% typical</p></div>',
        "</div></div></header>", '<div class="wrap">']

    # where to send a run — the practical answer, first
    rows = []
    for n_copies in COPY_COUNTS:
        ranked, no_room = [], 0
        for cls in classes:
            cell, cpus = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell:
                ranked.append((cell["total_steps_per_second"], cls, cell))
            elif any((cell_of(jobs, cls["name"], n, n_copies) or {}).get("status", "").startswith(
                    ("out_of_memory", "not_attempted_smaller")) for n in cpu_counts_for(cls)):
                no_room += 1
        if not ranked:
            continue
        ranked.sort(key=lambda r: -r[0])
        _, cls, cell = ranked[0]
        second = (f"{esc(ranked[1][1]['display_name'])}, "
                  f"{ranked[0][0] / ranked[1][0]:.2f}&times; as long"
                  if len(ranked) > 1 else "N/A")
        rows.append([f"{n_copies:,}", esc(cls["display_name"]),
                     f'<span class="mono">{esc(cls["name"])}</span>',
                     f"{10 * cell['hours_per_million_steps_per_copy']:.2f} h", second,
                     f'<span class="tag warn">{no_room}</span>' if no_room else "0"])
    out += ['<section><h2><span class="n">01</span> Where to send a run</h2>',
            "<h3>The fastest card at each size, and what it costs in wall time</h3>",
            '<p class="lede">Ten million steps per copy is the length of a real training run '
            "here, so the hours are quoted for that.</p>",
            table(["Copies", "Fastest card", "Node class", "Hours for 10M steps per copy",
                   "Next best card, and how much longer", "Cards that cannot hold it"], rows, numeric_from=3),
            "</section>"]

    # the per-copy-count tables
    out.append('<section><h2><span class="n">02</span> Every card, size by size</h2>'
               "<h3>Each class at whichever processor count ran fastest</h3>"
               '<p>Best value in colour, second best underlined; sorted by the aggregate rate. '
               "Both rates are shown because neither substitutes for the other &mdash; the "
               "aggregate says how much work the machine does, the per-copy rate how long any "
               "one copy takes to finish.</p>")
    for n_copies in COPY_COUNTS:
        measured = []
        for cls in classes:
            cell, cpus = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell:
                measured.append((cls, cell, cpus))
        if not measured:
            continue
        measured.sort(key=lambda r: -r[1]["total_steps_per_second"])
        totals = sorted({m[1]["total_steps_per_second"] for m in measured}, reverse=True)[:2]
        hours = sorted({m[1]["hours_per_million_steps_per_copy"] for m in measured})[:2]
        rows = [[f'<span class="mono">{esc(cls["name"])}</span>', esc(cls["display_name"]),
                 str(cpus), f"{c['seconds_per_iteration'] * 1000:.1f}",
                 mark(c["total_steps_per_second"] / 1e6, [t / 1e6 for t in totals], "{:.2f}"),
                 f"{c['steps_per_second_per_copy']:,.0f}",
                 mark(c["hours_per_million_steps_per_copy"], hours, "{:.3f}"),
                 f"{c['peak_device_memory_mb'] / 1000:.1f}",
                 f"{c['relative_spread_middle_half'] * 100:.2f}%"]
                for cls, c, cpus in measured]
        out += [f"<h3 style='margin-top:34px'>{n_copies:,} copies</h3>",
                table(["Node class", "Card", "Processors", "ms per iteration",
                       "Total M steps/s", "Steps/s per copy", "Hours per 1M steps/copy",
                       "Peak card memory (GB)", "Spread"], rows, numeric_from=2)]
    out.append("</section>")

    # both curves live in one section, so every section carries exactly one numbered heading
    figures = [(PLOTS / name, title, caption) for name, title, caption in (
        ("throughput_scaling.png", "How throughput scales with copies",
         "Left: the work the whole card does. Right: the rate one copy gets. The aggregate "
         "keeps climbing while each individual copy slows down, which is why both are shown."),
        ("peak_memory.png", "Memory the trainer actually touches",
         "Two bands, not a spread: every card at compute capability 8.0 and above sits on the "
         "lower line, every Turing and Pascal card on the upper one."))
        if (PLOTS / name).exists()]
    if figures:
        out.append('<section><h2><span class="n">03</span> Curves</h2>')
        for path, title, caption in figures:
            out.append(f"<h3 style='margin-top:28px'>{title}</h3>"
                       f'<figure><img alt="{esc(title)}" src="{data_uri(path)}">'
                       f"<figcaption>{caption}</figcaption></figure>")
        out.append("</section>")

    out += ['<section><h2><span class="n">04</span> Two findings</h2>',
            '<div class="finding"><h3>The processor count does not matter</h3>'
            "<p>Eight, sixteen and thirty-two processors give the same iteration time on every "
            "card measured &mdash; the slowest count trails the fastest by a fraction of a "
            "percent, with no consistent direction. The trainer keeps its arrays on the card "
            "and the host only dispatches. <strong>Ask for eight</strong>; the rest are free to "
            "carry other work.</p></div>",
            '<div class="finding"><h3>An older card spends about half as much memory again on '
            "the same run</h3>"
            f"<p>A thousand copies take {mem['older']:.2f}&nbsp;GB on every Turing and Pascal "
            f"card measured and {mem['newer']:.2f}&nbsp;GB on every Ampere, Ada, Hopper and "
            f"Blackwell card &mdash; {mem['ratio'] - 1:.0%} more on the older generation"
            + (f", and {mem['largest_older']:.1f}&nbsp;GB against "
               f"{mem['largest_newer']:.1f}&nbsp;GB at 4,096 copies, where it bites"
               if mem.get("largest_older") else "")
            + ". The split follows the card generation, not its size or speed, so it is the "
            "compiler emitting a different program &mdash; not the trainer asking for more. "
            "<strong>An older card reaches its ceiling sooner than its size suggests.</strong> "
            "What in that program costs the extra memory was not investigated here.</p></div>",
            "</section>"]

    # what is still missing, stated plainly
    measured_cards = {(c["display_name"], c["gpu_mem_mb"]) for c in classes
                      if any((c["name"], n) in jobs for n in cpu_counts_for(c))}
    missing = []
    for cls in classes:
        pending = [n for n in cpu_counts_for(cls) if (cls["name"], n) not in jobs]
        if not pending:
            continue
        covered = (cls["display_name"], cls["gpu_mem_mb"]) in measured_cards
        note = ("card already measured on another node class &mdash; the gap is the host "
                "processor only" if covered
                else '<span class="tag warn">this card is measured nowhere else</span>')
        missing.append(f'<li><span class="mono">{esc(cls["name"])}</span> '
                       f"({esc(cls['display_name'])}) at "
                       f"{', '.join(str(p) for p in pending)} processors &mdash; {note}</li>")
    out += ['<section><h2><span class="n">05</span> What is still missing</h2>']
    if missing:
        out += ["<h3>Jobs pinned to a busy node stay queued on purpose</h3>",
                "<p>They are collected when the node frees; nothing below has been abandoned.</p>",
                '<ul class="plain">'] + missing + ["</ul>"]
    else:
        out += ["<h3>Nothing</h3><p>Every node class reported every processor count.</p>"]
    out += ["</section>"]

    out += [f"<footer>Generated {now} from the run folder's own measurement files. "
            "Times are Pacific; the cluster's machines run Eastern and are converted where they "
            "are displayed. Every number excludes compilation and five warm-up iterations, and "
            "was taken only after the middle half of its timing rounds agreed to within 2% of "
            "their median.</footer>", "</div>"]
    return "\n".join(out)


def main():
    """Write the page beside the report."""
    classes, jobs = load_classes(), load_jobs()
    probes = [json.loads(p.read_text()) for p in sorted(PROBES.glob("*.json"))]
    page = RUN / "results.html"
    page.write_text(build(classes, jobs, probes))
    print(f"{page} written, {page.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
