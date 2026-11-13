"""Write a short diagnosis configuration from a campaign unit's configuration file (standard library only).

Usage: make_configuration.py <unit configuration .toml> <output .toml> <copies> <iterations> <window iterations>
The trainer, the environment and the learning-rate schedule stay those of the unit (the rate is still annealed
over the unit's own 2,048 iterations); only the number of policies, the length, the logging window and the
diagnostics switch change.
"""
import re
import sys


def changed(text, copies, iterations, window_iterations):
    """Return the unit's configuration text with the diagnosis values in place."""
    # before: "n_copies = 256", "copies_per_group = 256", "diagnostics = false", "iterations = 2048",
    #         "window_iterations = 8"
    # after:  "n_copies = 64", "copies_per_group = 64", "diagnostics = true", "iterations = 512",
    #         "window_iterations = 4"; "lr_anneal_iterations = 2048" is left as it is
    values = {"n_copies": copies, "copies_per_group": copies, "diagnostics": "true",
              "iterations": iterations, "window_iterations": window_iterations}
    for name, value in values.items():
        text, count = re.subn(rf"(?m)^{name} = .*$", f"{name} = {value}", text)
        if count != 1:
            raise ValueError(f"{name} appears {count} times in the unit's configuration, not once")
    return text


if __name__ == "__main__":
    source, output, copies, iterations, window_iterations = sys.argv[1:6]
    text = changed(open(source).read(), int(copies), int(iterations), int(window_iterations))
    open(output, "w").write("# Diagnosis configuration written by make_configuration.py from\n# "
                            + source.split("/")[-1] + "\n" + text)
