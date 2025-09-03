#!/bin/bash

# Uncertainty Comparison Experiment Launcher
# This script sets up and launches the complete uncertainty comparison experiment

set -e  # Exit on any error

echo "🚀 UNCERTAINTY COMPARISON EXPERIMENT SETUP"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f "01_uncertainty_comparison.py" ]]; then
    echo "❌ Error: Please run this script from the sweep_uncertainty directory"
    exit 1
fi

# Test the setup first
echo "🧪 Testing system setup..."
python test_setup.py

if [[ $? -ne 0 ]]; then
    echo "❌ Setup test failed! Please check the implementation."
    exit 1
fi

echo "✅ Setup test passed!"

# Test wall mask implementation
echo ""
echo "🏗️ Testing wall mask implementation..."
python test_wall_mask.py

if [[ $? -ne 0 ]]; then
    echo "❌ Wall mask test failed! Please check the maze structure."
    exit 1
fi

echo "✅ Wall mask test passed!"

# Create WandB sweep
echo ""
echo "📊 Creating WandB sweep..."
sweep_output=$(wandb sweep 01_wandb_sweep.yaml 2>&1)
sweep_id=$(echo "$sweep_output" | grep -o "uncertainty_comparison/[a-zA-Z0-9]*" | head -1)

if [[ -z "$sweep_id" ]]; then
    echo "❌ Failed to create WandB sweep. Output:"
    echo "$sweep_output"
    exit 1
fi

echo "✅ Sweep created: $sweep_id"

# Update batch script with actual sweep ID
echo "🔧 Updating batch script with sweep ID..."
sed -i "s/YOUR_SWEEP_ID/$sweep_id/g" slurm/00_batch_slurm.sh

# Make scripts executable
chmod +x slurm/*.sh
chmod +x slurm/*.slurm

echo "✅ Scripts configured!"

# Show next steps
echo ""
echo "🎯 EXPERIMENT READY TO LAUNCH!"
echo "==============================="
echo ""
echo "Your WandB sweep ID: $sweep_id"
echo ""
echo "To launch the experiments:"
echo "  ./slurm/00_batch_slurm.sh"
echo ""
echo "To monitor progress:"
echo "  wandb agent catresearch/$sweep_id     # Run single agent"
echo "  squeue -u \$USER                      # Check SLURM jobs"
echo "  tail -f slurm_output.log             # Watch logs"
echo ""
echo "Expected experiment duration: 12-24 hours"
echo "Expected number of sweep configs: ~9,000"
echo "  - RND: 20 architectures × 6 noise × 10 subsets × 5 seeds = 6,000"  
echo "  - RND-Linear: 5 dims × 6 noise × 10 subsets × 5 seeds = 1,500"
echo "  - Elliptical: 5 dims × 6 noise × 10 subsets × 5 seeds = 1,500"
echo "  - Each config averages over 10 internal runs automatically"
echo ""
echo "WandB dashboard: https://wandb.ai/catresearch/uncertainty_comparison"
echo ""
echo "🎉 Ready to go! Launch when ready."
