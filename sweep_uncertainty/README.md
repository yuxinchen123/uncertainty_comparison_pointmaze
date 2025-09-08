# Uncertainty Method Comparison on PointMaze

This project systematically compares three uncertainty estimation methods on the PointMaze-Large-v2 environment:

1. **RND (Random Network Distillation)**: Full neural network approach
2. **RND-Linear**: Simplified RND with shared linear features φ(s) 
3. **Elliptical Bonus**: Covariance-based uncertainty estimation

## Quick Start

1. **Initialize a WandB sweep:**
   ```bash
   wandb sweep 01_wandb_sweep.yaml
   ```
   This will output a sweep ID like `uncertainty_comparison/abc123def`

2. **Update the batch script:**
   Edit `slurm/00_batch_slurm.sh` and replace `YOUR_SWEEP_ID` with the actual sweep ID

3. **Launch experiments:**
   ```bash
   chmod +x slurm/00_batch_slurm.sh
   ./slurm/00_batch_slurm.sh
   ```

## Experiment Design

### Methods Compared
- **RND**: Neural networks with architectures [64,64], [128,128], [256,256], [512,512] and output dimensions [64, 128, 256, 512, 1024]
- **RND-Linear**: Linear feature layers φ(s) with dimensions [64, 128, 256, 512, 1024]
- **Elliptical**: Uses same φ(s) as RND-Linear via deterministic weight sharing

### Fair Evaluation Protocol
- **All methods use the same 10K samples** for training/covariance update
- **Each experiment runs 10 times** with the same data but different random seeds
- **Results are averaged** across these 10 runs to get stable estimates
- **Standard deviations** are reported to measure consistency

### Hyperparameter Sweep
- **RND Architectures**: 4 hidden layer configs × 5 output dimensions = 20 combinations
- **Linear Methods**: 5 φ(s) dimensions each (shared between RND-Linear and Elliptical)
- **Gaussian Noise**: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- **Data Subsets**: 10 different 10K samples (run_index 0-9)
- **Random Seeds**: 5 seeds for reproducibility
- **Averaging**: 10 runs per configuration for statistical stability
- **φ(s) Sharing**: 10 seeds shared for RND linear and Elliptical

### Evaluation Protocol
- **Ground Truth**: 1/√N(i) where N(i) is visit count per grid cell
- **Wall Exclusion**: Wall cells (maze_map[i][j]==1) completely excluded from evaluation
- **Metrics**: L2 distance and Pearson correlation on normalized uncertainties (open cells only)
- **Visualization**: Heatmaps logged to WandB for each run with wall overlay

## Key Features

### φ(s) Weight Sharing
RND-Linear and Elliptical methods share identical φ(s) feature weights using seedings:
- Enables fair comparison between linear methods
- WandB logs track φ(s) pairing information

### Training Strategies
- **All methods use identical 10K samples** for fair comparison
- **RND**:Training with neural networks
- **RND-Linear**: Full dataset training with linear features  
- **Elliptical**: Covariance update from full dataset
- **Averaging**: Each configuration repeated 10 times, results averaged for stability

### Regularization
- Subset training for RND-Linear (30% of data)
- Gaussian noise injection during training
- L1 uncertainty calculation (instead of L2) for final evaluation

## File Structure

```
sweep_uncertainty/
├── 01_uncertainty_comparison.py     # Main training script
├── 01_wandb_sweep.yaml             # WandB sweep configuration
├── utilities/
│   ├── __init__.py                 # Imports
│   ├── debug.py                    # Logging utilities
│   ├── environment.py              # PointMaze data loading
│   ├── uncertainty_methods.py      # RND/RND-Linear/Elliptical implementations
│   └── evaluation.py               # Metrics and visualization
└── slurm/
    ├── 00_batch_slurm.sh           # Batch job launcher
    ├── 01_run_gpu.slurm            # GPU SLURM script
    └── 02_run_cpu.slurm            # CPU SLURM script
```

## Expected Results

The sweep will generate comprehensive comparisons showing:
- Which uncertainty method best approximates ground truth 1/√N(i)
- How different noise levels affect each method
- Impact of architectural choices (network depth, feature dimensions)
- Consistency across different data subsets and random seeds

All results are logged to WandB with heatmap visualizations and quantitative metrics.

## Dependencies

- PyTorch
- Minari
- Gymnasium Robotics
- WandB
- NumPy
- Matplotlib
- tqdm
