# Ensemble-Based RND Uncertainty Methods

This directory contains ensemble-based implementations of Random Network Distillation (RND) methods, following the project proposal specifications.

## Overview

The ensemble-based methods extend the original RND approaches by using multiple predictors (K=10 by default) with bootstrap sampling. Each predictor samples its own dataset with replacement from the same 10k data pool, and ensemble uncertainty is calculated as the standard deviation across all predictors.

## Implemented Methods

### 1. Ensemble RND
- **Architecture**: K neural network predictors + 1 frozen target network
- **Training**: Bootstrap sampling with replacement for each predictor
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**: 
  - Shared frozen target network across all predictors
  - Independent predictor networks trained on bootstrap samples
  - Ensemble uncertainty = Std_{i=1,...,K}[e_i(s)]

### 2. Ensemble RND-Linear (SGD)
- **Architecture**: K linear predictors + frozen target vector θ̂
- **Training**: SGD training with bootstrap sampling
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**:
  - Shared frozen φ(s) feature transformation
  - Shared frozen target vector θ̂
  - Each predictor trained independently with SGD

### 3. Ensemble RND-Linear (LS)
- **Architecture**: K linear predictors + frozen target vector θ̂
- **Training**: Least squares fitting with bootstrap sampling
- **Uncertainty**: Standard deviation across K predictors
- **Key Features**:
  - Shared frozen φ(s) feature transformation
  - Shared frozen target vector θ̂
  - Each predictor fitted using least squares

## Project Structure

```
ensemble_based/
├── utilities/
│   ├── ensemble_uncertainty_methods.py  # Core ensemble implementations
│   ├── evaluation.py                     # Evaluation utilities (copied)
│   ├── environment.py                     # Dataset utilities (copied)
│   └── debug.py                          # Logging utilities (copied)
├── 01_ensemble_uncertainty_comparison.py  # Main comparison script
├── 02_ensemble_rnd_only.py               # Ensemble RND testing
├── 03_ensemble_rnd_linear_sgd.py          # Ensemble RND-Linear (SGD) testing
├── 04_ensemble_rnd_linear_ls.py           # Ensemble RND-Linear (LS) testing
├── 01_ensemble_wandb_sweep.yaml           # WandB sweep config (all methods)
├── 02_ensemble_rnd_wandb_sweep.yaml       # WandB sweep config (RND only)
├── 03_ensemble_rnd_linear_sgd_wandb_sweep.yaml  # WandB sweep config (SGD)
├── 04_ensemble_rnd_linear_ls_wandb_sweep.yaml   # WandB sweep config (LS)
├── slurm/                                 # SLURM batch scripts
└── README.md                             # This file
```

## Key Implementation Details

### Bootstrap Sampling
- Each predictor samples its own D_i with replacement from the 10k data pool
- All predictors use the same base dataset but get different bootstrap samples
- Sampling is done independently for each predictor

### Ensemble Uncertainty Calculation
- Per-model error: e_i(s) = ||f_theta_i(s) - f_hat_theta(s)||
- Ensemble uncertainty: r_int(s) = Std_{i=1,...,K}[e_i(s)]
- This captures epistemic uncertainty across the ensemble

### Shared Components
- **Target Network/Vector**: Frozen and shared across all predictors
- **Feature Transformation φ(s)**: Shared for RND-Linear methods
- **Training Data**: Same 10k pool, but each predictor gets its own bootstrap sample

## Usage

### Single Method Testing

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
    --num_epochs 30 \
    --wandb_switch False

# Test Ensemble RND-Linear (LS)
python 04_ensemble_rnd_linear_ls.py \
    --K 10 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --phi_dim 128 \
    --regularization 1e-6 \
    --wandb_switch False
```

### WandB Sweep Experiments

**1. Start WandB Sweep**
```bash
# All ensemble methods
wandb sweep 01_ensemble_wandb_sweep.yaml

# Individual methods
wandb sweep 02_ensemble_rnd_wandb_sweep.yaml
wandb sweep 03_ensemble_rnd_linear_sgd_wandb_sweep.yaml
wandb sweep 04_ensemble_rnd_linear_ls_wandb_sweep.yaml
```

**2. Configure SLURM Script**
Copy the sweep ID and paste it into `slurm/00_batch_slurm.sh`:
```bash
#!/bin/bash
job_id="your_sweep_id_here"
# ... rest of script
```

**3. Submit SLURM Jobs**
```bash
./slurm/00_batch_slurm.sh
```

## Key Parameters

### Ensemble Parameters
- `K`: Number of predictors in ensemble (default: 10)
- `num_samples`: Dataset size (default: 10000)
- `num_averaging_runs`: Number of averaging runs (default: 10)

### RND Parameters
- `output_dim`: RND output dimension (default: 128)
- `hidden_dims`: Hidden layer dimensions (default: "128,128")

### RND-Linear Parameters
- `phi_dim`: Feature dimension (default: 128)
- `phi_seed`: Seed for shared φ(s) weights (default: 42)
- `regularization`: Regularization for least squares (default: 1e-6)

### Training Parameters
- `num_epochs`: Training epochs (default: 30)
- `gaussian_noise`: Noise level (default: 0.0)

## Expected Results

### Performance Characteristics
- **Ensemble RND**: Neural network flexibility with ensemble uncertainty
- **Ensemble RND-Linear (SGD)**: Fast SGD training with ensemble benefits
- **Ensemble RND-Linear (LS)**: Fastest training with analytical solution

### Uncertainty Quality
- Ensemble methods should provide more robust uncertainty estimates
- Bootstrap sampling captures data uncertainty
- Standard deviation across predictors captures model uncertainty

## Comparison with Original Methods

The ensemble-based methods maintain the same evaluation protocol as the original methods:
- Same 10k dataset sampling
- Same ground truth calculation (1/√N(i))
- Same L2 distance evaluation metric
- Same visualization and normalization

The key differences are:
- **Sampling**: Bootstrap sampling with replacement for each predictor
- **Uncertainty**: Ensemble standard deviation instead of single-model error
- **Training**: K independent predictors instead of single predictor

## Requirements

Same as the original RND project:
```bash
pip install torch torchvision
pip install gymnasium gymnasium-robotics
pip install minari
pip install wandb
pip install numpy matplotlib
pip install argparse
```





