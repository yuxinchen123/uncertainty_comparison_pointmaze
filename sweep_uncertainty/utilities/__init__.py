# Import only the functions that actually exist
from .debug import print_or_wandb_log
from .environment import load_pointmaze_dataset, extract_positions_from_dataset, get_maze_map
from .uncertainty_methods import (
    RNDMethod, 
    RNDLinearMethod,  # Alias for RNDLinearLSMethod (backward compatibility)
    RNDLinearLSMethod, 
    RNDLinearSGDMethod,
    EllipticalBonusMethod
)
from .evaluation import (
    calculate_ground_truth, evaluate_uncertainty_method,
    normalize_uncertainty_matrix, compute_l2_distance,
    compute_correlation, save_heatmap_to_wandb
)