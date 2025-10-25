# RND Uncertainty Method Comparison on PointMaze

This repository contains a systematic comparison of uncertainty estimation methods on PointMaze environments, focusing on Random Network Distillation (RND) and related approaches.

## Project Overview

A comprehensive evaluation framework comparing uncertainty estimation methods:

**Single-Model Methods:**
- **RND**: Neural network-based Random Network Distillation
- **RND-Linear**: Linear feature-based approach with SGD training or least squares fitting
- **Elliptical Bonus**: Covariance-based uncertainty estimation

**Ensemble-Based Methods:**
- **Ensemble RND**: Multiple neural network predictors with bootstrap sampling
- **Ensemble RND-Linear (SGD)**: Multiple linear predictors with SGD training
- **Ensemble RND-Linear (LS)**: Multiple linear predictors with least squares fitting

All methods are evaluated on PointMaze-Large-v2 using identical data subsets and ground truth calculations for fair comparison.

## Project Structure

```
RND/
├── sweep_uncertainty/                      # Main experiment directory
│   ├── 01_uncertainty_comparison.py        # Main training/evaluation script
│   ├── 01_wandb_sweep.yaml                # WandB sweep configuration
│   ├── 02_elliptical_only.py              # Focused elliptical method evaluation
│   ├── 02_elliptical_only.yaml            # Elliptical-only sweep configuration
│   ├── 03_rnd_linear_only.py              # RND-Linear focused evaluation
│   ├── 03_rnd_linear_only.yaml            # RND-Linear sweep configuration
│   ├── 04_rnd_linear_sgd_sweep.py         # RND-Linear SGD hyperparameter sweep
│   ├── 04_rnd_linear_sgd.yaml             # SGD hyperparameter sweep configuration
│   ├── utilities/                          # Core implementations
│   │   ├── uncertainty_methods.py          # Method implementations
│   │   ├── evaluation.py                   # Ground truth & metrics
│   │   ├── evaluation_unnormalized.py      # Raw uncertainty evaluation
│   │   ├── environment.py                  # Dataset utilities
│   │   └── debug.py                        # Logging utilities
│   └── slurm/                              # SLURM batch scripts
├── ensemble_based/                         # Ensemble-based RND methods
│   ├── 01_ensemble_uncertainty_comparison.py  # Main ensemble comparison script
│   ├── 02_ensemble_rnd_only.py               # Ensemble RND testing
│   ├── 03_ensemble_rnd_linear_sgd.py          # Ensemble RND-Linear (SGD) testing
│   ├── 04_ensemble_rnd_linear_ls.py           # Ensemble RND-Linear (LS) testing
│   ├── 01_ensemble_wandb_sweep.yaml           # WandB sweep config (all methods)
│   ├── 02_ensemble_rnd_wandb_sweep.yaml       # WandB sweep config (RND only)
│   ├── 03_ensemble_rnd_linear_sgd_wandb_sweep.yaml  # WandB sweep config (SGD)
│   ├── 04_ensemble_rnd_linear_ls_wandb_sweep.yaml   # WandB sweep config (LS)
│   ├── utilities/                          # Core ensemble implementations
│   │   ├── ensemble_uncertainty_methods.py  # Ensemble method implementations
│   │   ├── evaluation.py                   # Evaluation utilities (copied)
│   │   ├── environment.py                  # Dataset utilities (copied)
│   │   └── debug.py                        # Logging utilities (copied)
│   ├── slurm/                              # SLURM batch scripts
│   └── README.md                           # Ensemble methods documentation
├── saved_data/                             # Experimental results
└── README.md                               # This file
```

## Method Implementations

### 1. RND (Random Network Distillation)
- **Architecture**: Configurable neural networks (target + predictor)
- **Training**: Standard gradient descent with MSE loss
- **Uncertainty**: L1 distance between target and predictor outputs
- **Parameters**: `hidden_dims`, `output_dim`, `num_epochs`, `gaussian_noise`

### 2. RND-Linear (Linear RND with Least Squares)
- **Features**: Shared φ(s) linear transformation (2D → feature_dim)
- **Target**: φ(s)ᵀθ where θ is random vector
- **Training**: **Direct least squares solution** (no gradient descent)
- **Uncertainty**: |φ(s)ᵀθ - predictor(φ(s))|
- **Parameters**: `phi_dim`, `phi_seed`, `gaussian_noise`

### 3. Elliptical Bonus
- **Features**: Same shared φ(s) as RND-Linear
- **Method**: (Corrected) covariance matrix estimation from training data
- **Uncertainty**: √(φ(s)ᵀ Σ⁻¹ φ(s)) with regularization
- **Parameters**: `phi_dim`, `phi_seed`
- **Note**: No noise parameters (analytical method)

## Ensemble-Based Method Implementations

### 4. Ensemble RND
- **Architecture**: K neural network predictors + 1 frozen target network
- **Training**: Bootstrap sampling with replacement for each predictor
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**: 
  - Shared frozen target network across all predictors
  - Independent predictor networks trained on bootstrap samples
  - Ensemble uncertainty = Std_{i=1,...,K}[e_i(s)]

### 5. Ensemble RND-Linear (SGD)
- **Architecture**: K linear predictors + frozen target vector θ̂
- **Training**: SGD training with bootstrap sampling
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**:
  - Shared frozen φ(s) feature transformation
  - Shared frozen target vector θ̂
  - Each predictor trained independently with SGD

### 6. Ensemble RND-Linear (LS)
- **Architecture**: K linear predictors + frozen target vector θ̂
- **Training**: Least squares fitting with bootstrap sampling
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**:
  - Shared frozen φ(s) feature transformation
  - Shared frozen target vector θ̂
  - Each predictor fitted using least squares

## Experimental Protocol

### Data Handling
1. **Dataset**: PointMaze-Large-v2 from D4RL/Minari
2. **Sampling**: 10K random positions per run (configurable)
3. **Seeding**: Controlled random sampling for reproducibility
4. **Coordinate System**: Exact mapping from validated notebook (X[-6,6], Y[-4.5,4.5] → 9×12 grid)

### Ground Truth Calculation
- **Method**: 1/√N(i) where N(i) is visit count per grid cell
- **Exclusions**: Wall cells excluded from all calculations
- **Source**: Same 10K data subset used for method training

### Evaluation Metrics
- **Primary**: L2 distance between normalized uncertainty matrices
- **Secondary**: Pearson correlation coefficient
- **Visualization**: Heatmaps with wall masking

### Averaging Protocol
- **Per Config**: 10 averaging runs with different 10K subsets
- **Data Consistency**: Same subset used for ground truth and method training within each run
- **Final Result**: Mean ± standard deviation across 10 runs

## Usage

### Single Method Testing
```bash
cd sweep_uncertainty

# Test RND
python 01_uncertainty_comparison.py \
    --method rnd \
    --output_dim 128 \
    --hidden_dims "128,128" \
    --num_epochs 30 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --wandb_switch False

# Test RND-Linear (with SGD)
python 03_rnd_linear_only.py \
    --method rnd_linear \
    --phi_dim 128 \
    --phi_seed 42 \
    --num_epochs 20 \
    --gaussian_noise 0.5 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --wandb_switch False

# Test Elliptical Bonus (corrected)
python 02_elliptical_only.py \
    --method elliptical \
    --phi_dim 128 \
    --phi_seed 42 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --wandb_switch False
```

### Ensemble Method Testing
```bash
cd ensemble_based

# Test Ensemble RND
python 02_ensemble_rnd_only.py \
    --K 10 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --output_dim 128 \
    --hidden_dims "128,128" \
    --num_epochs 30 \
    --wandb_switch False

# Test Ensemble RND-Linear (SGD)
python 03_ensemble_rnd_linear_sgd.py \
    --K 10 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --phi_dim 128 \
    --phi_seed 42 \
    --num_epochs 20 \
    --gaussian_noise 0.5 \
    --wandb_switch False

# Test Ensemble RND-Linear (LS)
python 04_ensemble_rnd_linear_ls.py \
    --K 10 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --phi_dim 128 \
    --phi_seed 42 \
    --gaussian_noise 0.5 \
    --wandb_switch False
```

### WandB Slurm Sweep Experiments


**1. Start WandB Sweep**
```bash
wandb sweep 01_wandb_sweep.yaml
```
or
```bash
wandb sweep 02_elliptical_only.yaml
```
or
```bash
wandb sweep 03_rnd_linear_only.yaml
```
or for ensemble methods:
```bash
wandb sweep 01_ensemble_wandb_sweep.yaml
```
or
```bash
wandb sweep 02_ensemble_rnd_wandb_sweep.yaml
```
or
```bash
wandb sweep 03_ensemble_rnd_linear_sgd_wandb_sweep.yaml
```
or
```bash
wandb sweep 04_ensemble_rnd_linear_ls_wandb_sweep.yaml
```

You will get output like:
```
wandb: Creating sweep from: 01_wandb_sweep.yaml
wandb: Created sweep with ID: nbkcxcge
wandb: View sweep at: https://wandb.ai/catresearch/robust/sweeps/nbkcxcge
wandb: Run sweep agent with: wandb agent catresearch/robust/nbkcxcge
```

**2. Configure SLURM Script**

Copy the sweep ID (e.g., `robust/nbkcxcge`) and paste it into line 4 of `slurm/00_batch_slurm.sh`:

```bash
#!/bin/bash

# Define job_id variable with your sweep ID
job_id="robust/nbkcxcge"

# for i in $(seq 1 1); do
#     sbatch slurm/01_run_gpu.slurm $job_id&
# done
```
**3. Submit SLURM Jobs**

For initial testing (submit 1 job):
Run
```bash
./slurm/00_batch_slurm.sh
```

Keep the loop commented for testing:
```bash
# for i in $(seq 1 1); do
#     sbatch slurm/01_run_gpu.slurm $job_id&
# done
```

**4. Scale Up (After Testing)**

Once verified, modify the script to submit multiple jobs:

```bash
# GPU jobs
for i in $(seq 1 5); do
    sbatch slurm/01_run_gpu.slurm $job_id&
done

# CPU jobs (no GPU limit)
for i in $(seq 1 5); do
    sbatch slurm/02_run_gnolim.slurm $job_id&
done

# Additional CPU jobs
for i in $(seq 1 5); do
    sbatch slurm/03_run_cpu.slurm $job_id&
done
```

**5. Monitor Progress**
```bash
# Check job status
squeue -u $USER

## Configuration Parameters


### Key Parameters
- `num_samples`: 10000 (data subset size)
- `num_averaging_runs`: 10 (runs per config)
- `grid_rows`: 9, `grid_cols`: 12 (PointMaze grid)
- `phi_seed`: Shared feature weight seed for RND-Linear and Elliptical

## Key Features

### Robust Evaluation
- ✅ **Statistical Significance**: 10-run averaging with confidence intervals
- ✅ **Reproducible**: Controlled seeding for all random components
- ✅ **Normalized**: Proper uncertainty matrix normalization before comparison

### Implementation Improvements
- ✅ **Least Squares**: RND-Linear uses direct solution (faster, more stable)
- ✅ **Regularization**: Elliptical method with numerical stability improvements


## Requirements

```bash
pip install torch torchvision
pip install gymnasium gymnasium-robotics
pip install minari
pip install wandb
pip install numpy matplotlib
pip install argparse
```

## Expected Results

### Performance Characteristics
- **RND**: Neural network flexibility, requires hyperparameter tuning
- **RND-Linear**: Fast least squares fitting, fewer parameters
- **Elliptical**: Analytical solution, no training require


