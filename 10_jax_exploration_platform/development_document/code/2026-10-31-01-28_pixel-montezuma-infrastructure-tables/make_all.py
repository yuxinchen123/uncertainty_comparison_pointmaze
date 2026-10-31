"""Generate the pixel study's tables from its specifications and registered cases.

The default command only formats metadata. Statistical checks and plots require
--analyze on a compute host, and use only result paths in study_registry.json.
No unregistered resource probe is a formal measurement in this report.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import re
import socket
import sys
import textwrap

import report_source as source


HERE = Path(__file__).resolve().parent
DOCUMENT = HERE.parents[1] / "RND_JAX_exploration_development_document.tex"
PLATFORM = HERE.parents[2]
sys.path.insert(0, str(PLATFORM / "scripts" / "atari_pixel_campaign"))
from controller import investigated_at_counts
from ideas import REQUIRED_HARDWARE, investigated_hypotheses, measured_hypotheses, summarize_pairs

HARDWARE_NAMES = {"h100": "H100", "jaguar03_rtx_a4500": "Jaguar03 RTX A4500"}
AGGREGATE = HERE / "data" / "registered_measurements.json"
PLOT_DIRECTORY = HERE / "plots"
FIGURE_MARKER = "pixel-montezuma-paired-measurements"
MAX_PANEL_MECHANISMS = 10
MAX_PANEL_LABEL_LINES = 20


def read_json(path):
    """Read complete records, retaining malformed input as an explicit failure."""
    return json.loads(Path(path).read_text())


def digest(value):
    """Identify the exact metadata and measurements used for an output."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def latex_text(value):
    """Escape arbitrary case names and status strings without changing their wording."""
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def flat_yaml(path):
    """Read the campaign's flat scalar settings without importing a training package."""
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or ":" not in line:
            raise ValueError("resolved configuration must remain a flat scalar mapping")
        name, raw = line.split(":", 1)
        raw = raw.strip()
        if raw in {"true", "false"}:
            value = raw == "true"
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
        result[name] = value
    return result


def check_fixed_settings(campaign):
    """Stop if the report's reviewed fixed settings differ from the resolved run."""
    resolved = flat_yaml(campaign / "config_resolved.yaml")
    differences = {name: (expected, resolved.get(name))
                   for name, expected in source.RESOLVED_SETTINGS.items()
                   if resolved.get(name) != expected}
    if differences:
        raise ValueError(f"review the report specification before updating changed settings: {differences}")
    manifest = (campaign / "manifest.yaml").read_text()
    if source.CLEANRL_COMMIT not in manifest or source.JAXTARI_COMMIT not in manifest:
        raise ValueError("the report and campaign must name the same pinned source commits")


def entries(registry):
    """Visit specified and chosen adaptive mechanisms, keeping the unchosen slots explicit."""
    return registry["initial_ideas"] + [entry for batch in registry["adaptive_batches"]
                                         for entry in batch["hypotheses"]]


def registered_cases(campaign, registry):
    """Read only indexed immutable case attempts; never search probe directories."""
    records, seen = [], set()
    for entry in entries(registry):
        for hardware, summaries in entry.get("results", {}).items():
            for summary in summaries:
                path = Path(summary["result_path"])
                path = path if path.is_absolute() else campaign / path
                path = path.resolve(strict=True)
                if not path.is_relative_to(campaign.resolve()):
                    raise ValueError(f"registered result is outside this campaign: {path}")
                if path in seen:
                    raise ValueError(f"duplicate registered result: {path}")
                seen.add(path)
                result = read_json(path)
                if result.get("idea") != entry["name"] or result.get("hardware_class") != hardware:
                    raise ValueError(f"result identity disagrees with the registry: {path}")
                for name in ("decision", "measurement_valid", "correctness_passed", "copies",
                             "median_speedup", "paired_rounds", "measurement_scope"):
                    if name in summary and result.get(name) != summary[name]:
                        raise ValueError(f"result field {name} disagrees with the registry: {path}")
                records.append({"registry_summary": summary, "result": result,
                                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def fingerprint(registry, records):
    """Invalidate a statistical report when any indexed input or specification changes."""
    return digest({"registry": registry,
                   "results": [(row["path"], row["sha256"]) for row in records],
                   "report_specification": hashlib.sha256((HERE / "report_source.py").read_bytes()).hexdigest()})


def require_compute():
    """Keep numerical result analysis and plotting off the login node."""
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") and host not in {"serval02", "serval05", "jaguar03"}:
        raise SystemExit("--analyze and --plots require a compute allocation or an approved direct GPU host")


def analyze(registry, records, input_fingerprint):
    """Audit stored paired decisions with the campaign's own selection function."""
    require_compute()
    audits = []
    for record in records:
        result = record["result"]
        if not result.get("rounds"):
            continue
        checked = summarize_pairs(result["rounds"], correctness_passed=result.get("correctness_passed", False),
                                  contaminated=result.get("contaminated", False))
        for name in ("decision", "measurement_valid", "paired_rounds"):
            if checked[name] != result.get(name):
                raise ValueError(f"paired-decision audit failed for {record['path']}: {name}")
        audits.append({"path": record["path"], "selection_audit": checked})
    aggregate = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "input_fingerprint": input_fingerprint, "reference_commit": registry["reference_commit"],
                 "records": records, "paired_decision_audits": audits}
    AGGREGATE.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    return aggregate


def category_table(rows, compared=False):
    """Use the document's grey row rules and black category separators."""
    if compared:
        lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.25cm}?",
                 r">{\raggedright\arraybackslash}p{4.75cm}?",
                 r">{\raggedright\arraybackslash}p{\dimexpr\textwidth-8cm-4\tabcolsep-2\arrayrulewidth\relax}@{}}",
                 r"\blackrule", r"\textbf{Hyperparameter} & \textbf{Original CleanRL} & \textbf{Pixel JAX learner} \\"]
        span = r"\multicolumn{2}{>{\raggedright\arraybackslash}p{\dimexpr\textwidth-3.25cm-2\tabcolsep-\arrayrulewidth\relax}@{}}"
        columns = 3
    else:
        lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.25cm}?",
                 r">{\raggedright\arraybackslash}p{\dimexpr\textwidth-3.25cm-2\tabcolsep-\arrayrulewidth\relax}@{}}",
                 r"\blackrule", r"\textbf{Parameter} & \textbf{Pixel environment or implementation} \\"]
        span, columns = "", 2
    for category, parameter, value in rows:
        if category:
            lines.extend([r"\blackrule", rf"\multicolumn{{{columns}}}{{@{{}}l}}{{\textit{{{category}}}}} \\"])
        lines.append(r"\greyrule")
        rendered = f"{span}{{{value}}}" if compared else value
        lines.append(f"{parameter} & {rendered} " + r"\\")
    return "\n".join(lines + [r"\blackrule", r"\end{tabular}"])


def two_column_table(rows, first="Study record", second="Current status"):
    """Format short status records without shrinking text or hiding missing results."""
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.3cm}?",
             r">{\raggedright\arraybackslash}p{\dimexpr\textwidth-5.3cm-2\tabcolsep-\arrayrulewidth\relax}@{}}",
             r"\blackrule", rf"\textbf{{{first}}} & \textbf{{{second}}} \\"]
    for parameter, value in rows:
        lines.extend([r"\greyrule", f"{parameter} & {value} " + r"\\"])
    return "\n".join(lines + [r"\blackrule", r"\end{tabular}"])


def status_table(registry, state):
    """Use the controller's definitions of investigation and complete copy-count coverage."""
    specified = registry["initial_ideas"]
    initial_names = {entry["name"] for entry in specified}
    adaptive = [entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"]]
    investigated = set(investigated_hypotheses(registry))
    measured = set(measured_hypotheses(registry))
    counts = registry.get("initial_matched_copy_counts", [])
    complete_count = (sum(investigated_at_counts(registry, [name], counts) for name in initial_names)
                      if counts else 0)
    rows = [("specified mechanisms", str(len(specified))),
            ("specified mechanisms investigated on both GPU types", f"{len(initial_names & investigated)} of {len(specified)}"),
            ("specified mechanisms timed correctly on both GPU types", f"{len(initial_names & measured)} of {len(specified)}"),
            ("specified mechanisms covering every required copy count", f"{complete_count} of {len(specified)}" if counts else "required copy counts not yet recorded"),
            ("additional mechanisms chosen from profiles", f"{sum(bool(entry['name']) for entry in adaptive)} of {len(adaptive)}"),
            ("additional mechanisms investigated on both GPU types", f"{len(investigated - initial_names)} of {len(adaptive)}"),
            ("controller phase", latex_text(str(state.get("phase", "not recorded")).replace("_", " ")))]
    for queue_state, count in sorted(state.get("queue_counts", {}).items()):
        rows.append((f"registered queue cases {latex_text(queue_state)}", str(count)))
    return two_column_table(rows)


def outcome(record):
    """Keep numerical rejection, timing rejection, and invalid timing separate."""
    result, summary = record["result"], record["registry_summary"]
    if result.get("failure_stage") == "correctness" and result.get("correctness_attempted"):
        return "correctness rejected"
    if not (result.get("measurement_valid") and result.get("correctness_passed")
            and summary.get("implementation_distinct") and not result.get("contaminated")):
        return "invalid measurement"
    return {"accepted": "accepted timing", "inconclusive": "inconclusive timing",
            "rejected": "rejected timing"}.get(result.get("decision"), "invalid measurement")


def outcomes_table(records):
    """Report all registered outcomes, without treating a failed test as zero speed."""
    categories = ("accepted timing", "inconclusive timing", "rejected timing",
                  "correctness rejected", "invalid measurement")
    counts = {hardware: Counter(outcome(row) for row in records
                                if row["result"]["hardware_class"] == hardware)
              for hardware in REQUIRED_HARDWARE}
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.4cm}?r?r@{}}", r"\blackrule",
             r"\textbf{Registered case outcome} & \textbf{H100} & \textbf{Jaguar03 RTX A4500} \\"]
    for category in categories:
        lines.extend([r"\greyrule", latex_text(category) + " & "
                      + " & ".join(str(counts[hardware][category]) for hardware in REQUIRED_HARDWARE) + r" \\"])
    return "\n".join(lines + [r"\blackrule", r"\end{tabular}"])


def write_ledger(registry, records, input_fingerprint, analyzed):
    """Preserve every formal result, including rejected and inconclusive attempts."""
    lines = ["# Pixel Montezuma infrastructure measurements", "",
             "Generated by `make_all.py`; edit `report_source.py` or the campaign records.", "",
             f"Input fingerprint: `{input_fingerprint}`.", "",
             f"Statistical analysis: {'current compute aggregate' if analyzed else 'pending; metadata only'}.", "",
             "Resource probes outside the study registry are excluded.", ""]
    if not records:
        lines.append("No paired case result has been registered. Measurements are in progress.")
    for number, row in enumerate(records, 1):
        result = row["result"]
        lines.extend([f"## Registered attempt {number}: {result['idea'].replace('_', ' ')}", "",
                      f"- Hardware: {HARDWARE_NAMES[result['hardware_class']]}; host: {result.get('hostname', 'not recorded')}.",
                      f"- Independent policy copies: {result.get('copies', 'not recorded')}.",
                      f"- Outcome: {outcome(row)}.",
                      f"- Measurement scope: {result.get('measurement_scope', 'not timed').replace('_', ' ')}.",
                      f"- Source commit: `{result.get('source_commit', 'not recorded')}`.",
                      f"- Result: [{Path(row['path']).name}]({row['path']})."])
        if analyzed and result.get("measurement_valid"):
            lines.extend([f"- Median paired speed ratio: {result.get('median_speedup', 'not recorded')}.",
                          "- Candidate aggregate transitions per second: "
                          f"{result.get('median_candidate_transitions_per_second_total', 'not recorded')}."])
        if result.get("error"):
            lines.extend(["", "Failure details:", "", "```text", str(result["error"]), "```"])
        lines.append("")
    (HERE / "analysis.md").write_text("\n".join(lines).rstrip() + "\n")


def plot_specifications(records, input_fingerprint, plot_directory=PLOT_DIRECTORY):
    """Describe bounded panels without importing plotting or numerical packages."""
    grouped = defaultdict(list)
    for row in records:
        result = row["result"]
        if outcome(row) in {"invalid measurement", "correctness rejected"}:
            continue
        cohort = (result["hardware_class"], result.get("hostname", "not recorded"), result["copies"],
                  result["source_commit"], result.get("measurement_scope", "training_iteration_execution"),
                  digest(result["reference"]))
        grouped[cohort].append(row)
    panels = []
    for cohort, rows in sorted(grouped.items()):
        # Before: one unbounded cohort; after: complete rows in panels of at most ten.
        chunks, chunk, lines = [], [], 0
        for row in rows:
            label = textwrap.fill(row["result"]["idea"].replace("_", " "), width=40,
                                  break_long_words=False, break_on_hyphens=False)
            label_lines = len(label.splitlines())
            if label_lines > MAX_PANEL_LABEL_LINES:
                raise ValueError("a mechanism label exceeds the readable panel height")
            if chunk and (len(chunk) == MAX_PANEL_MECHANISMS
                          or lines + label_lines > MAX_PANEL_LABEL_LINES):
                chunks.append(chunk)
                chunk, lines = [], 0
            chunk.append((row, label))
            lines += label_lines
        if chunk:
            chunks.append(chunk)
        for index, chunk in enumerate(chunks, 1):
            hardware, host, copies, commit, scope, reference = cohort
            stem = f"paired-speed-ratios_{digest(cohort)}_panel-{index:02d}-of-{len(chunks):02d}"
            panels.append({"schema_version": 2, "path": str(Path(plot_directory).resolve() / (stem + ".pdf")),
                "hardware": hardware, "host": host, "copies": copies, "source_commit": commit,
                "scope": scope, "reference_fingerprint": reference, "panel_index": index,
                "panel_count": len(chunks), "mechanism_names": [row["result"]["idea"] for row, _ in chunk],
                "wrapped_labels": [label for _, label in chunk],
                "results": [{"path": row["path"], "sha256": row["sha256"]} for row, _ in chunk],
                "input_fingerprint": input_fingerprint})
    return panels


def make_plots(records, input_fingerprint):
    """Draw the bounded panel specification on compute and record every generated PDF."""
    require_compute()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    generated = plot_specifications(records, input_fingerprint)
    indexed = {row["path"]: row["result"] for row in records}
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for panel in generated:
        results = [indexed[row["path"]] for row in panel["results"]]
        label_lines = sum(len(label.splitlines()) for label in panel["wrapped_labels"])
        figure, axis = plt.subplots(figsize=(8.6, max(3.2, 1.8 + 0.25 * label_lines + 0.12 * len(results))))
        for position, result in enumerate(results):
            ratios = result["paired_speedups"]
            color = {"accepted": "#2166ac", "inconclusive": "#b8860b", "rejected": "#555555"}[result["decision"]]
            axis.scatter(ratios, [position] * len(ratios), s=15, alpha=0.6, color=color)
            axis.scatter([result["median_speedup"]], [position], marker="D", s=30, color=color)
        axis.set_yticks(range(len(results)), panel["wrapped_labels"], fontsize=12.5)
        axis.invert_yaxis()
        axis.axvline(1.0, color="black", linewidth=0.8)
        axis.set_xlabel("Reference duration / candidate duration (larger is faster)", fontsize=12.5)
        axis.tick_params(axis="x", labelsize=12)
        copy_noun = "copy" if panel["copies"] == 1 else "copies"
        axis.set_title(f"{HARDWARE_NAMES[panel['hardware']]}, {panel['copies']} independent policy {copy_noun}, "
                       f"{panel['host']}\n{panel['scope'].replace('_', ' ')}; "
                       f"panel {panel['panel_index']} of {panel['panel_count']}", fontsize=13)
        colors = {"accepted": "#2166ac", "inconclusive": "#b8860b", "rejected": "#555555"}
        present = {result["decision"] for result in results}
        handles = [Line2D([], [], marker="o", linestyle="none", color=colors[name], label=name)
                   for name in colors if name in present]
        handles.append(Line2D([], [], marker="D", linestyle="none", color="black", label="paired median"))
        axis.legend(handles=handles, loc="best", fontsize=12)
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout(pad=1.2)
        for suffix in ("pdf", "png"):
            figure.savefig(Path(panel["path"]).with_suffix("." + suffix), dpi=180)
        plt.close(figure)
    (PLOT_DIRECTORY / "plot_manifest.json").write_text(json.dumps(generated, indent=2) + "\n")


def figure_source(manifest_path, expected_panels, document):
    """Include every current panel exclusively, rejecting stale fingerprints or missing files."""
    manifest_path = Path(manifest_path)
    if not expected_panels and not manifest_path.exists():
        return "% No registered valid paired timing is available for a figure."
    if not manifest_path.exists() or read_json(manifest_path) != expected_panels:
        raise ValueError("plots require a current manifest with matching input fingerprint and panels; run --plots on compute")
    figures = []
    for index, panel in enumerate(expected_panels):
        path = Path(panel["path"])
        if not path.is_file() or path.suffix != ".pdf":
            raise ValueError(f"the current plot manifest names a missing PDF: {path}")
        relative = os.path.relpath(path, Path(document).resolve().parent)
        copies = panel["copies"]
        caption = (f"Paired execution-speed measurements on {HARDWARE_NAMES[panel['hardware']]} "
                   f"({panel['host']}), with {copies} independent policy "
                   + ("copy" if copies == 1 else "copies")
                   + f". Source commit: \\texttt{{{latex_text(panel['source_commit'])}}}. "
                   + f"Measurement scope: {latex_text(panel['scope'].replace('_', ' '))}. "
                   + f"Reference configuration hash: \\texttt{{{panel['reference_fingerprint']}}}. "
                   + f"Panel {panel['panel_index']} of {panel['panel_count']} for this configuration. "
                   + "Each circle is one paired reference-to-candidate duration ratio; the diamond is its median. "
                   + "Ratios above one favor the candidate. Each panel has its own horizontal range. "
                   + "These execution timings do not establish learning returns, larger-copy performance, "
                   + "or performance of the final combination.")
        label = "fig:pixel-montezuma-paired-" + path.stem.removeprefix("paired-speed-ratios_")
        block = [r"\begin{figure}[H]", r"\centering",
                 rf"\includegraphics[width=\textwidth]{{\detokenize{{{relative}}}}}",
                 "\\caption{" + caption + "}", "\\label{" + label + "}"]
        if index == 0:
            block.append(r"\label{fig:pixel-montezuma-background-cache-paired}")
        figures.append("\n".join(block + [r"\end{figure}"]))
    return "\n\n".join(figures) if figures else "% No registered valid paired timing is available for a figure."


def splice(document, replacements, figures=None):
    """Change only uniquely marked table blocks and the separate generated figure block."""
    text = document.read_text()
    groups = [("TABLE", replacements)]
    if figures is not None:
        groups.append(("FIGURES", {FIGURE_MARKER: figures}))
    for kind, blocks in groups:
        for name, content in blocks.items():
            start = f"% >>> AUTO-GENERATED {kind} START: {name}"
            end = f"% <<< AUTO-GENERATED {kind} END: {name}"
            if text.count(start) != 1 or text.count(end) != 1:
                raise ValueError(f"expected exactly one marker pair for {name}")
            expression = re.compile(re.escape(start) + r"\n.*?\n" + re.escape(end), re.DOTALL)
            text, count = expression.subn(lambda _: start + "\n" + content + "\n" + end, text)
            if count != 1:
                raise ValueError(f"could not replace {kind.lower()} block {name}")
    document.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=source.CAMPAIGN_ROOT)
    parser.add_argument("--document", type=Path, default=DOCUMENT)
    parser.add_argument("--analyze", action="store_true", help="audit paired statistics on compute")
    parser.add_argument("--plots", action="store_true", help="plot current compute-audited paired results")
    arguments = parser.parse_args()
    campaign = arguments.campaign.resolve(strict=True)
    check_fixed_settings(campaign)
    registry = read_json(campaign / "study_registry.json")
    state_path = campaign / "campaign_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    records = registered_cases(campaign, registry)
    input_fingerprint = fingerprint(registry, records)
    aggregate = analyze(registry, records, input_fingerprint) if arguments.analyze else None
    if records and aggregate is None:
        if not AGGREGATE.exists() or read_json(AGGREGATE).get("input_fingerprint") != input_fingerprint:
            raise SystemExit("new registered measurements require --analyze on compute before updating the report")
        aggregate = read_json(AGGREGATE)
    if arguments.plots:
        if aggregate is None and records:
            raise SystemExit("plotting requires a current compute aggregate")
        make_plots(records, input_fingerprint)
    figures = figure_source(PLOT_DIRECTORY / "plot_manifest.json",
                            plot_specifications(records, input_fingerprint), arguments.document)
    splice(arguments.document, {
        "pixel-montezuma-environment": category_table(source.ENVIRONMENT_ROWS),
        "pixel-montezuma-learner": category_table(source.LEARNER_ROWS, compared=True),
        "pixel-montezuma-networks": category_table(source.NETWORK_ROWS),
        "pixel-montezuma-study-status": status_table(registry, state),
        "pixel-montezuma-case-outcomes": outcomes_table(records),
    }, figures=figures)
    write_ledger(registry, records, input_fingerprint, aggregate is not None)
    print(f"Updated five table blocks and all figure panels from {len(records)} registered paired case attempts.")


if __name__ == "__main__":
    main()
