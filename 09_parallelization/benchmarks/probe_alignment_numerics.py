"""Does padding the parameter windows change what a multiplication computes?

The alignment change moves every parameter to a sixteen-byte boundary, which makes the library
select a different kernel — its four-numbers-at-a-time one instead of its scalar-load one. A
different kernel adds the same products in a different order, so the last bits of the result can
differ. This measures by how much, on the trainer's real shapes, with the SAME numbers in both
cases: one weight tensor read from an aligned window of a buffer and the same numbers read from a
window offset by two, which is the misalignment the packed layout produced.

Whole-iteration comparisons cannot answer this question: a difference in the last bits of the
first action changes the trajectory, so after one iteration the two runs are simply doing
different things. This isolates the arithmetic.

Usage (through the H100 lock wrapper): python probe_alignment_numerics.py --n-copies 1024
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"

# the layers of one update step, as (name, rows, in, out)
LAYERS = [("actor and critic first layer, packed", 128, 4, 128),
          ("actor second layer", 128, 64, 64),
          ("RND predictor first layer", 128, 4, 256),
          ("RND predictor second layer", 128, 256, 128),
          ("RND predictor third layer", 128, 128, 128)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-tf32", action="store_true",
                    help="run in full single precision, to separate a difference caused by the "
                         "reduced-precision matrix units from one caused by anything else")
    args = ap.parse_args()
    # "high" selects the reduced-precision matrix units, which is what the trainer runs with;
    # "highest" keeps full single precision, and running both says whether an observed
    # difference is the reduced precision the configuration already asks for
    torch.set_float32_matmul_precision("highest" if args.no_tf32 else "high")
    C = args.n_copies
    dev = torch.device("cuda")
    rows = []

    for name, m, k, n in LAYERS:
        # two buffers holding the same weights: in the first the window starts at a multiple of
        # four numbers, in the second two numbers later, which is where the packed layout put
        # them. Two buffers rather than one, because writing both windows into one buffer would
        # have the second write overwrite most of the first.
        # before: window at number 0 of its buffer; after: window at number 2 of its buffer
        block = k * n
        w = torch.randn(C, k, n, device=dev)
        buf_a = torch.zeros(C, block + 4, device=dev)
        buf_o = torch.zeros(C, block + 4, device=dev)
        buf_a[:, 0:block] = w.reshape(C, block)
        buf_o[:, 2:2 + block] = w.reshape(C, block)
        aligned = buf_a[:, 0:block].view(C, k, n)
        offset = buf_o[:, 2:2 + block].view(C, k, n)
        assert torch.equal(aligned, offset), "the two windows do not hold the same numbers"
        assert aligned.data_ptr() % 16 == 0 and offset.data_ptr() % 16 != 0, \
            "the probe did not actually produce one aligned and one unaligned window"

        x = torch.randn(C, m, k, device=dev)
        out_a = torch.bmm(x, aligned)
        out_o = torch.bmm(x, offset)
        d = (out_a - out_o).abs().max().item()
        scale = max(out_a.abs().max().item(), 1e-12)
        rows.append({"layer": name, "rows": m, "in": k, "out": n,
                     "worst_absolute": d, "worst_relative": d / scale,
                     "largest_output": scale})
        print(f"  {name:>42s}: worst absolute {d:.3e}, relative {d/scale:.3e}, "
              f"largest output {scale:.3e}")
        del buf_a, buf_o, w, x, out_a, out_o
        torch.cuda.empty_cache()

    worst = max(r["worst_relative"] for r in rows)
    print(f"\nworst relative difference over all layers: {worst:.3e}")
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_alignment_numerics"
                   f"_C{C}{args.tag}.json")
    p.write_text(json.dumps({
        "n_copies": C, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "matrix_units": "full single precision" if args.no_tf32 else "reduced precision (TF32)",
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "worst_relative": worst, "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
