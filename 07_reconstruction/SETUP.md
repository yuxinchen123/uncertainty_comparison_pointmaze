# Setup: Run 07_reconstruction on Another Computer

Use this to clone the environment and run the training code on a new machine.

## 1. Clone the repo

```bash
git clone <your-repo-url> RND
cd RND
```

If you use submodules (e.g. RLeXplore), run:

```bash
git submodule update --init --recursive
```

## 2. Python version

- Use **Python 3.11** (the project's conda env is `exploration`, Python 3.11).
- Check: `python3 --version`

## 3. Create a virtual environment (recommended)

**Option A – venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

**Option B – conda:**

```bash
conda create -n exploration python=3.11 -y
conda activate exploration
```

## 4. Install dependencies

Install the package itself (editable, src-layout) — this pulls the dependencies from `pyproject.toml` and makes
`import rnd_exploration` resolve from any directory:

```bash
pip install -e 07_reconstruction
```

For **GPU (CUDA)** install PyTorch with the right CUDA version, then the rest:

```bash
# Example for CUDA 11.8 (adjust for your driver/CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r 07_reconstruction/requirements.txt
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for your OS/CUDA combo.

## 5. Run the training entry point

The entry point is `train.py`. Because the package is installed editable (step 4), `import rnd_exploration`
resolves regardless of the working directory; run `train.py` from `07_reconstruction`:

```bash
cd 07_reconstruction
python train.py --help
```

Example local run (no wandb, CPU, short):

```bash
cd 07_reconstruction
python train.py --algorithm rnd_linear_next_state --use_wandb false --device cpu --total_timesteps 1000
```

## 6. Optional: Weights & Biases (wandb)

If you use wandb sweeps (e.g. `wandb agent ...`):

```bash
pip install wandb
wandb login
```

Set the sweep config and run as in your current setup.

## 7. Quick checklist

| Step | Command / check |
|------|------------------|
| Python 3.8+ | `python3 --version` |
| Venv/conda | `source .venv/bin/activate` or `conda activate exploration` |
| Package | `pip install -e 07_reconstruction` (editable; pulls deps from pyproject.toml) |
| CWD | `cd 07_reconstruction` before running `train.py` |
| GPU (optional) | Install `torch` with CUDA, then use `--device cuda` |

## 8. Legacy precursors

The numbered precursors `01_gt.py`, `02_rnd_rlexplore.py`, `03_rnd_my_implementation.py` now live under
`legacy/` as a frozen reference and are **not** runnable against the current package (their imports target the
pre-reorganization layout). To run them, check out the pre-reorganization commit — see `legacy/README.md`.
