# Random Network Distillation (RND) Uncertainty Analysis

This repository contains code for systematic comparison of uncertainty estimation methods on PointMaze environments using Random Network Distillation (RND) and related approaches.

## Project Structure

```
RND/
├── sweep_uncertainty/           # Main uncertainty comparison experiment
│   ├── 01_uncertainty_comparison.py    # Main training script
│   ├── 01_wandb_sweep.yaml            # WandB sweep configuration
│   ├── utilities/                      # Core utility functions
│   │   ├── evaluation.py               # Evaluation metrics and ground truth
│   │   ├── uncertainty_methods.py      # Uncertainty method implementations
│   │   └── environment.py              # Environment utilities
│   └── slurm/                          # SLURM batch job scripts
├── pointmaze_count.ipynb        # Reference notebook with validated methods
└── saved_data/                  # Saved experimental data
```

## Methods Compared

1. **RND (Random Network Distillation)**: Neural network-based uncertainty estimation
2. **RND-Linear**: Linear feature-based uncertainty with shared φ(s) representations
3. **Elliptical Bonus**: Covariance-based uncertainty estimation

## Key Features

- **Fair Comparison**: All methods use identical 10K random data subsets for training
- **Ground Truth**: Calculates uncertainty as 1/√N(i) from visit counts
- **10-Run Averaging**: Each experiment averages results over 10 different random subsets
- **Coordinate Consistency**: Uses exact grid mapping from validated Jupyter notebook
- **Wall Exclusion**: Properly handles maze walls in evaluation and visualization

## Usage

### Single Method Test
```bash
cd sweep_uncertainty
python 01_uncertainty_comparison.py --method rnd --output_dim 128 --hidden_dims "128,128"
```

### WandB Sweep
```bash
cd sweep_uncertainty
wandb sweep 01_wandb_sweep.yaml
wandb agent your-username/uncertainty_comparison/your-sweep-id
```

### SLURM Batch Jobs
```bash
cd sweep_uncertainty/slurm
./00_batch_slurm.sh
```

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Minari
- WandB
- NumPy
- Matplotlib

## Experimental Protocol

1. **Data Sampling**: 10K samples randomly drawn from PointMaze-Large-v2 dataset
2. **Training**: Each method trains on the same data subset
3. **Ground Truth**: Calculated from visit counts in the same subset
4. **Evaluation**: L2 distance between predicted and ground truth uncertainty
5. **Averaging**: Results averaged over 10 different random subsets

## Key Metrics

- **L2 Distance**: Primary metric for uncertainty prediction accuracy
- **Correlation**: Secondary metric for uncertainty ranking
- **Visualization**: Heatmaps showing uncertainty distributions with wall exclusion

## Notes

- Ensures φ(s) weight sharing between RND-Linear and Elliptical methods
- Uses deterministic seeding for reproducible random subsets
- Implements exact coordinate mapping from reference notebook
- Excludes maze walls from all calculations and visualizations
