"""Print the rows of one or more benchmark result JSONs as a small table.

A reading aid for the ledger work: every result file has the same row shape (copy count,
seconds per iteration, aggregate environment steps per second, peak memory), so one printer
serves torch, jax and the cpu benchmarks.
"""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def show(path):
    """Print one result file's identity line and one line per measured copy count."""
    d = json.loads(Path(path).read_text())
    print(f"=== {Path(path).name}  impl={d.get('impl')} git={d.get('git')} "
          f"torch={d.get('torch', '')}{d.get('jax', '')}")
    for r in d.get("rows", []):
        # a torch row calls it sec_per_iteration, a jax row sec_per_iteration_median
        sec = r.get("sec_per_iteration") or r.get("sec_per_iteration_median")
        print(f"  style={r.get('style','?'):>16s} timing={r.get('timing','?'):>9s} "
              f"C={r['n_copies']:>5d}  {sec*1e3:9.3f} ms  "
              f"{r.get('env_steps_per_sec', 0)/1e6:7.3f} M steps/s  "
              f"{r.get('peak_vram_mb', 0):9.1f} MB")
    for f in d.get("failures", []):
        print(f"  FAILED C={f['n_copies']}: {f['error'][:160]}")


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        for p in sorted(RESULTS.glob(pat)):
            show(p)
