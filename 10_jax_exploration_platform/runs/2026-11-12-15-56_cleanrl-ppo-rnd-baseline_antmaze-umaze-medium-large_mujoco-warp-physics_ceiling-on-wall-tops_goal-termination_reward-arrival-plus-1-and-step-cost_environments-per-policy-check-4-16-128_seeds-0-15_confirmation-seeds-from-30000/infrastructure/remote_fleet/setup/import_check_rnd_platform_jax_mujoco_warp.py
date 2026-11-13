"""Import check of the PointMaze and AntMaze study's environment on a remote server: imports only, no training.

    <env>/bin/python -I import_check_rnd_platform_jax_mujoco_warp.py processor   # a processor-only compute node
    <env>/bin/python -I import_check_rnd_platform_jax_mujoco_warp.py card        # a job that holds one card

Run it inside a Slurm job, never on a login node. Both modes import what the trainer imports for an AntMaze unit
with the MuJoCo Warp physics (jax, mujoco, mujoco.mjx with its Warp implementation, warp, flax, optax, jaxatari)
and print each version. `card` also lists the devices jax and warp see and requires exactly one card. Nothing is
compiled and no model is built: it is not a speed or memory test. The last line is IMPORT_CHECK_PASSED or the
script ends with an error.
"""
import importlib.metadata
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "processor"
if MODE not in ("processor", "card"):
    raise SystemExit("mode is processor or card")
PACKAGES = ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "mujoco", "mujoco-mjx", "warp-lang", "flax", "optax",
            "jaxatari")
began = time.time()
print("python", sys.version.split()[0], "at", sys.executable)
for package in PACKAGES:
    print(f"{package}=={importlib.metadata.version(package)}")

# the imports of the trainer's MuJoCo Warp module (src/exploration_platform/envs/antmaze/warp_antmaze.py)
import jax  # noqa: E402
import mujoco  # noqa: E402
from mujoco import mjx  # noqa: E402,F401
from mujoco.mjx import warp as mjx_warp  # noqa: E402
from mujoco.mjx.warp import types as mjx_warp_types  # noqa: E402,F401
import warp as wp  # noqa: E402
import flax  # noqa: E402,F401
import optax  # noqa: E402,F401
import jaxatari  # noqa: E402,F401

# before: mujoco-mjx without warp-lang beside it; after: WARP_INSTALLED False and mujoco_warp None
print("mujoco.mjx Warp implementation: warp installed", mjx_warp.WARP_INSTALLED,
      "| mujoco_warp module loaded", mjx_warp.mujoco_warp is not None)
if not (mjx_warp.WARP_INSTALLED and mjx_warp.mujoco_warp is not None):
    raise SystemExit("mujoco.mjx cannot use its Warp implementation in this environment")
print(f"imports took {time.time() - began:.1f} s")

if MODE == "card":
    # jax: one card and the card back end
    devices = jax.devices()
    print("jax default back end:", jax.default_backend(), "| jax devices:", devices)
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise SystemExit("jax does not see exactly one card")
    # warp: initialised here only, so that a processor-only node never reaches it
    wp.init()
    cards = wp.get_cuda_devices()
    # warp's own start line above names the CUDA toolkit and driver versions
    print("warp CUDA devices:", [(str(card), card.name, card.arch) for card in cards])
    if len(cards) != 1:
        raise SystemExit("warp does not see exactly one card")
print("IMPORT_CHECK_PASSED", MODE)
