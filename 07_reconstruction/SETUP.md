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

- Use **Python 3.8–3.11** (3.10 is a safe choice).
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
conda create -n rnd python=3.10 -y
conda activate rnd
```

## 4. Install dependencies

From the **project root** (where this `07_reconstruction` folder lives):

```bash
pip install -r 07_reconstruction/requirements.txt
```

For **GPU (CUDA)** install PyTorch with the right CUDA version, then the rest:

```bash
# Example for CUDA 11.8 (adjust for your driver/CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r 07_reconstruction/requirements.txt
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for your OS/CUDA combo.

## 5. Run from the right directory

Scripts expect to be run with `07_reconstruction` as the current working directory (so imports like `utilities`, `env_wrapper`, `intrinsic` resolve).

```bash
cd 07_reconstruction
python 03_rnd_my_implementation.py --help
```

Example local run (no wandb, CPU, short):

```bash
cd 07_reconstruction
python 03_rnd_my_implementation.py --use_wandb false --device cpu --total_timesteps 1000
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
| Venv/conda | `source .venv/bin/activate` or `conda activate rnd` |
| Deps | `pip install -r 07_reconstruction/requirements.txt` |
| CWD | `cd 07_reconstruction` before running scripts |
| GPU (optional) | Install `torch` with CUDA, then use `--device cuda` |

## 8. If you also run other scripts in this repo

- `02_rnd_rlexplore.py` uses **rllte** (RLeXplore). Install from the RLeXplore submodule or `pip install rllte` if needed.
- `01_gt.py` and `03_rnd_my_implementation.py` only need the packages in `07_reconstruction/requirements.txt`.
