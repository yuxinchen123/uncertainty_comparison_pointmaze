"""Compare the first lines of a new run's metrics.jsonl with an archived run's, field by field.

    python compare_metric_prefix.py ARCHIVED_METRICS NEW_METRICS OUTPUT_JSON [--lines N]

Standard library only. Line i of the new file is compared with line i of the archived file: every
field the archived line has, except `unix` (the wall-clock time of the write), must be in the new
line with an equal value. Fields only the new line has (for example the game events) are listed
and do not count as a difference. N defaults to the number of lines of the new file; the archived
file is read only that far. The result is written to OUTPUT_JSON and printed; the exit status is 1
when a field differs or a file is shorter than N lines.

The values are the JSON numbers train_final.py wrote. Python writes a float with the shortest text
that reads back to the same float, so equal parsed values mean equal bits; two NaN values count as
equal.
"""

import argparse
import json
import math
from pathlib import Path

IGNORED_FIELDS = ("unix",)
REPORTED_DIFFERENCES = 20


def same_value(archived, new):
    """Equality of two parsed JSON values, with NaN equal to NaN and lists compared item by item."""
    # before: [1.5, [true, NaN]] against [1.5, [true, NaN]] ; after: True
    # before: 1 against 1.0 ; after: True (JSON does not tell 1 from 1.0 apart reliably)
    if isinstance(archived, list) or isinstance(new, list):
        return (isinstance(archived, list) and isinstance(new, list) and len(archived) == len(new)
                and all(same_value(left, right) for left, right in zip(archived, new)))
    if isinstance(archived, float) and isinstance(new, float) and math.isnan(archived) and math.isnan(new):
        return True
    if isinstance(archived, bool) != isinstance(new, bool):
        return False
    return archived == new


def read_lines(path, count):
    """The first `count` non-empty lines of a metrics file, parsed (all of them when count is None)."""
    lines = []
    with path.open() as stream:
        for text in stream:
            if count is not None and len(lines) == count:
                break
            if text.strip():
                lines.append(json.loads(text))
    return lines


def compare(archived_lines, new_lines, wanted):
    """The comparison record of two lists of parsed metric lines over the first `wanted` lines."""
    differences, extra_fields, fields_compared = [], set(), 0
    for index, (archived, new) in enumerate(zip(archived_lines, new_lines), start=1):
        extra_fields |= set(new) - set(archived)
        for field, value in archived.items():
            if field in IGNORED_FIELDS:
                continue
            fields_compared += 1
            if field not in new:
                differences.append({"line": index, "field": field, "archived": value, "new": "(field missing)"})
            elif not same_value(value, new[field]):
                differences.append({"line": index, "field": field, "archived": value, "new": new[field]})
    lines_compared = min(len(archived_lines), len(new_lines))
    return {"lines_wanted": wanted, "archived_lines_read": len(archived_lines), "new_lines_read": len(new_lines),
            "lines_compared": lines_compared, "fields_compared": fields_compared,
            "differing_fields": len(differences),
            "first_differences": differences[:REPORTED_DIFFERENCES],
            "fields_only_in_new_run": sorted(extra_fields),
            "passed": not differences and lines_compared == wanted and wanted > 0}


def main(argv=None):
    """Compare two metrics files and write the record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archived", type=Path, help="the archived run's metrics.jsonl")
    parser.add_argument("new", type=Path, help="the new run's metrics.jsonl")
    parser.add_argument("output", type=Path, help="where the comparison record is written")
    parser.add_argument("--lines", type=int, default=None, help="lines to compare (default: all of the new file)")
    args = parser.parse_args(argv)
    new_lines = read_lines(args.new, args.lines)
    wanted = args.lines if args.lines is not None else len(new_lines)
    record = {"archived": str(args.archived), "new": str(args.new),
              **compare(read_lines(args.archived, wanted), new_lines, wanted)}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
