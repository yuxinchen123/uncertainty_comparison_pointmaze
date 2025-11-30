# Import only the functions that actually exist
from .debug import print_or_wandb_log
from .environment import load_pointmaze_dataset, extract_positions_from_dataset, get_maze_map
from .evaluation import (
    calculate_ground_truth, evaluate_uncertainty_method,
    normalize_uncertainty_matrix, compute_l2_distance,
    compute_min_c_l1_norm_diff, compute_min_c_l2_norm_diff,
    compute_min_c_l1_norm_inv, compute_min_c_l2_norm_inv,
    save_heatmap_to_wandb, get_phi_weights
)

