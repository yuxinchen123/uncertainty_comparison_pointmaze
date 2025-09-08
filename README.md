# RND Uncertainty Method Comparison on PointMaze

This repository contains a systematic comparison of uncertainty estimation methods on PointMaze environments, focusing on Random Network Distillation (RND) and related approaches.

## Project Overview

A comprehensive evaluation framework comparing three uncertainty estimation methods:
- **RND**: Neural network-based Random Network Distillation
- **RND-Linear**: Linear feature-based approach with least squares fitting
- **Elliptical Bonus**: Covariance-based uncertainty estimation

All methods are evaluated on PointMaze-Large-v2 using identical data subsets and ground truth calculations for fair comparison.

## Project Structure

```
RND/
├── sweep_uncertainty/                  # Main experiment directory
│   ├── 01_uncertainty_comparison.py    # Main training/evaluation script
│   ├── 01_wandb_sweep.yaml            # WandB sweep configuration
│   ├── utilities/                      # Core implementations
│   │   ├── uncertainty_methods.py      # Method implementations
│   │   ├── evaluation.py               # Ground truth & metrics
│   │   ├── environment.py              # Dataset utilities
│   │   └── debug.py                    # Logging utilities
│   └── slurm/                          # SLURM batch scripts
├── pointmaze_count.ipynb               # Reference validation notebook
├── saved_data/                         # Experimental results
└── README.md                           # This file
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
- **Method**: Covariance matrix estimation from training data
- **Uncertainty**: √(φ(s)ᵀ Σ⁻¹ φ(s)) with regularization
- **Parameters**: `phi_dim`, `phi_seed`
- **Note**: No noise parameters (analytical method)

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

# Test RND-Linear (with least squares)
python 01_uncertainty_comparison.py \
    --method rnd_linear \
    --phi_dim 128 \
    --phi_seed 42 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --wandb_switch False

# Test Elliptical Bonus
python 01_uncertainty_comparison.py \
    --method elliptical \
    --phi_dim 128 \
    --phi_seed 42 \
    --num_samples 10000 \
    --num_averaging_runs 10 \
    --wandb_switch False
```

### WandB Sweep Experiments
To run the program locally
1. cd /p/rlprojects/RND/sweep_uncertainty && python 01_uncertainty_comparison.py

To run the program with slurm

1. Start sweep
wandb sweep 01_wandb_sweep.yaml

You get return (for example):

wandb: Creating sweep from: 01_wandb_sweep.yaml
wandb: Created sweep with ID: nbkcxcge
wandb: View sweep at: https://wandb.ai/catresearch/robust/sweeps/nbkcxcge
wandb: Run sweep agent with: wandb agent catresearch/robust/nbkcxcge


2. Setup slurm folder
copy  
"robust/nbkcxcge"

to the line 4 of 
slurm/00_batch_slurm.sh


Your script looks like 

------------------------start---------------------------------------
#!/bin/bash

#Define two variables with name job_id and cpu_core
job_id="robust/nbkcxcge"

# for i in $(seq 1 1); do
#     sbatch slurm/01_run_gpu.slurm $job_id&
# done

------------------------end---------------------------------------



3. Submit job

Run
./slurm/00_batch_slurm.sh 



Only run submit 1 job for test, 

# for i in $(seq 1 1); do
#     sbatch slurm/01_run_gpu.slurm $job_id&
# done

If it looks good, modify the script to submit multiple


------------------------start---------------------------------------
for i in $(seq 1 5); do
    sbatch slurm/01_run_gpu.slurm $job_id&
done

for i in $(seq 1 5); do
    sbatch slurm/02_run_gnolim.slurm $job_id&
done

for i in $(seq 1 5); do
    sbatch slurm/03_run_cpu.slurm $job_id&
done


------------------------end---------------------------------------

## Configuration Parameters

### Sweep Configuration (`01_wandb_sweep.yaml`)
- **Methods**: `['rnd', 'rnd_linear', 'elliptical']`
- **Architecture**: Multiple hidden dimensions and output dimensions
- **φ(s) Sharing**: Deterministic feature weights for fair comparison
- **Noise Levels**: Gaussian noise variations (0.0 to 10.0)
- **Seeds**: Multiple random seeds for robust evaluation
- **Averaging**: 10 runs per configuration

### Key Parameters
- `num_samples`: 10000 (data subset size)
- `num_averaging_runs`: 10 (runs per config)
- `grid_rows`: 9, `grid_cols`: 12 (PointMaze grid)
- `phi_seed`: Shared feature weight seed for RND-Linear and Elliptical

## Key Features

### Fair Comparison Design
- ✅ **Identical Data**: All methods use same 10K subset within each run
- ✅ **Shared Features**: RND-Linear and Elliptical use identical φ(s) weights
- ✅ **Ground Truth Consistency**: Same data for uncertainty calculation
- ✅ **Wall Handling**: Proper maze wall exclusion throughout

### Robust Evaluation
- ✅ **Statistical Significance**: 10-run averaging with confidence intervals
- ✅ **Reproducible**: Controlled seeding for all random components
- ✅ **Validated**: Coordinate mapping verified against reference notebook
- ✅ **Normalized**: Proper uncertainty matrix normalization before comparison

### Implementation Improvements
- ✅ **Least Squares**: RND-Linear uses direct solution (faster, more stable)
- ✅ **Regularization**: Elliptical method with numerical stability improvements
- ✅ **Debugging**: Comprehensive error checking and validation

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
- **Elliptical**: Analytical solution, no training required

### Typical Metrics
- **L2 Distance**: 0.5-2.0 (lower is better)
- **Correlation**: 0.3-0.8 (higher is better)
- **Runtime**: Elliptical < RND-Linear < RND

## Notes and Limitations

### Design Decisions
- Uses exact coordinate mapping from validated notebook implementation
- Excludes gaussian noise from Elliptical method (analytical approach)
- Implements direct least squares for RND-Linear (no gradient descent)
- Applies proper regularization for numerical stability

### Known Issues
- Requires sufficient memory for full dataset loading
- CUDA setup needed for GPU acceleration
- WandB account required for sweep experiments

### Future Improvements
- [ ] Add more uncertainty methods (e.g., MC Dropout)
- [ ] Extend to other environments beyond PointMaze
- [ ] Implement online uncertainty estimation
- [ ] Add uncertainty calibration metrics
