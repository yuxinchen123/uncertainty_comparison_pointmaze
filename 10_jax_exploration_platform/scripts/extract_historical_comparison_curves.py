"""Read the original historical curve accessors once and cache the 11 comparison curves."""
import argparse
import json
import sys
from pathlib import Path

from write_corrected_report import FIRST, SECOND


def main():
    """Keep historical seed errors and record the source of each fixed setting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = args.run_dir.parent
    first = next(runs.glob("2026-08-16-22-47_*extension-of-first-batch"))
    second = next(runs.glob("2026-08-17-10-45_*extension-2-of-first-batch"))
    sys.path.insert(0, str(second / "analysis/code"))
    from curves_all import curves_all
    cache = args.run_dir / "historical_curves.json"
    if cache.exists():
        previous = json.loads(cache.read_text())
        if "first" in previous["curves"]:
            print(f"using {cache}", flush=True)
            return
        raise ValueError("historical cache must contain separate curves for both figures")
    first_available = curves_all(first)
    second_available = curves_all(second)
    curves, sources = {"first": {}, "second": {}}, {}
    for source, settings, copies in ((first, FIRST[:4], 1024),
                                    (second, list(dict.fromkeys(FIRST[4:] + SECOND)), 128)):
        available = first_available if source == first else second_available
        for bonus, rate, weight in settings:
            # The old no-bonus control used intrinsic-advantage weight one.
            old_weight = 1. if bonus == "none" else weight
            curve = available[(bonus, rate, old_weight)]
            if not curve["env_steps_per_copy"] or curve["env_steps_per_copy"][-1] < 1e9:
                raise ValueError(f"incomplete historical curve: {source}, {bonus}, {rate}, {weight}")
            key = f"{bonus}:{rate:g}:{weight:g}"
            sources[key] = {"run_dir": str(source), "copies": copies,
                            "historical_intrinsic_weight": old_weight}
        print(f"read {len(settings)} selected curves from {source.name}", flush=True)
    for i, (b, r, w) in enumerate(FIRST):
        available = first_available if i < 4 else second_available
        curves["first"][f"{b}:{r:g}:{w:g}"] = available[(b, r, 1. if b == "none" else w)]
    for b, r, w in SECOND:
        curves["second"][f"{b}:{r:g}:{w:g}"] = second_available[(b, r, w)]
    if len(sources) != 11:
        raise ValueError(f"expected 11 historical settings, found {len(sources)}")
    output = args.run_dir / "historical_curves.json"
    output.write_text(json.dumps({"accessor": str(second / "analysis/code/curves_all.py"),
                                  "sources": sources, "curves": curves,
                                  "first_figure_sources": {
                                      f"{b}:{r:g}:{w:g}": {"run_dir": str(first if i < 4 else second),
                                                           "copies": 1024 if i < 4 else 128}
                                      for i, (b, r, w) in enumerate(FIRST)}}, allow_nan=False))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
