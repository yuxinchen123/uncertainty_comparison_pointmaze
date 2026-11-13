"""Write a short diagnosis configuration from a campaign unit's configuration file (standard library only).

Usage: make_configuration.py <unit configuration .toml> <output .toml> <copies> <iterations> <window iterations>
                             [<intrinsic weight>,<intrinsic weight>,...]
The trainer, the environment and the learning-rate schedule stay those of the unit (the rate is still annealed
over the unit's own 2,048 iterations); only the number of policies, the length, the logging window and the
diagnostics switch change. With a list of intrinsic weights the run holds one configuration per weight, each
with <copies> policies on the same seeds, and every other setting as in the unit.
"""
import re
import sys


def with_weights(text, weights):
    """Return the configuration text with one configuration per intrinsic weight."""
    # before: "parameter_settings = [[0.006, 0.01, 4.0, 0.1]]" and weights [0.25, 1.0]
    # after:  "parameter_settings = [[0.006, 0.01, 0.25, 0.1], [0.006, 0.01, 1.0, 0.1]]"
    found = re.search(r"(?m)^parameter_settings = \[\[(.*)\]\]$", text)
    if found is None or "]" in found.group(1):
        raise ValueError("the unit's configuration must hold exactly one configuration")
    rate, entropy, _, clip = [part.strip() for part in found.group(1).split(",")]
    cells = ", ".join(f"[{rate}, {entropy}, {weight}, {clip}]" for weight in weights)
    return text.replace(found.group(0), f"parameter_settings = [{cells}]")


def changed(text, copies, iterations, window_iterations, cells=1):
    """Return the unit's configuration text with the diagnosis values in place."""
    # before: "n_copies = 256", "copies_per_group = 256", "diagnostics = false", "iterations = 2048",
    #         "window_iterations = 8"
    # after:  "n_copies = 64", "copies_per_group = 64", "diagnostics = true", "iterations = 512",
    #         "window_iterations = 4"; "lr_anneal_iterations = 2048" is left as it is
    values = {"n_copies": copies * cells, "copies_per_group": copies, "diagnostics": "true",
              "iterations": iterations, "window_iterations": window_iterations}
    for name, value in values.items():
        text, count = re.subn(rf"(?m)^{name} = .*$", f"{name} = {value}", text)
        if count != 1:
            raise ValueError(f"{name} appears {count} times in the unit's configuration, not once")
    return text


if __name__ == "__main__":
    source, output, copies, iterations, window_iterations = sys.argv[1:6]
    weights = [float(w) for w in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
    text = open(source).read()
    if weights:
        text = with_weights(text, weights)
    text = changed(text, int(copies), int(iterations), int(window_iterations), len(weights) if weights else 1)
    open(output, "w").write("# Diagnosis configuration written by make_configuration.py from\n# "
                            + source.split("/")[-1] + "\n" + text)
