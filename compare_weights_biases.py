import torch
import numpy as np
import sys
sys.path.append('sweep_uncertainty')
from utilities.uncertainty_methods import RNDLinearSGDMethod, RNDLinearLSMethod
from utilities.evaluation import get_maze_map

def create_phi_weights_deterministic(feature_dim, seed):
    """Create deterministic phi weights"""
    torch.manual_seed(seed)
    phi_net = torch.nn.Linear(2, feature_dim)
    return phi_net.state_dict()

print("=" * 100)
print("WEIGHTS AND BIASES COMPARISON: SGD_1000 vs SGD_2000 vs OLS")
print("=" * 100)

device = 'cpu'
feature_dim = 256
phi_seed = 42
theta_seed = 42

# Create shared phi weights
phi_weights = create_phi_weights_deterministic(feature_dim, phi_seed)

# Generate positions
np.random.seed(42)
positions = np.random.randn(10000, 2) * 3

# Train SGD_1000
print("\nTraining SGD_1000...")
torch.manual_seed(42)
sgd_1000 = RNDLinearSGDMethod(
    feature_dim, device=device,
    phi_weights=phi_weights,
    theta_seed=theta_seed,
    predictor_seed=42
)
sgd_1000.train_on_positions(positions, num_epochs=1000, gaussian_noise=0.0)

# Extract weights and bias for SGD_1000
sgd_1000_weights = sgd_1000.predictor.weight.data.squeeze().cpu().numpy()  # [256]
sgd_1000_bias = sgd_1000.predictor.bias.data.item()

# Train SGD_2000 (continue from SGD_1000 state)
print("\nTraining SGD_2000 (continuing from SGD_1000)...")
sgd_2000 = RNDLinearSGDMethod(
    feature_dim, device=device,
    phi_weights=phi_weights,
    theta_seed=theta_seed,
    predictor_seed=42
)
# Copy weights from SGD_1000 to continue training
sgd_2000.predictor.weight.data = sgd_1000.predictor.weight.data.clone()
sgd_2000.predictor.bias.data = sgd_1000.predictor.bias.data.clone()
sgd_2000.optimizer = torch.optim.Adam(sgd_2000.predictor.parameters(), lr=0.001)
sgd_2000.train_on_positions(positions, num_epochs=2000, gaussian_noise=0.0)

# Extract weights and bias for SGD_2000
sgd_2000_weights = sgd_2000.predictor.weight.data.squeeze().cpu().numpy()  # [256]
sgd_2000_bias = sgd_2000.predictor.bias.data.item()

# Train OLS
print("\nTraining OLS...")
ols = RNDLinearLSMethod(
    feature_dim, device=device,
    phi_weights=phi_weights,
    theta_seed=theta_seed
)
ols.train_on_positions(positions, num_epochs=1, gaussian_noise=0.0)

# Extract weights and bias for OLS
ols_weights = ols.predictor_weights.cpu().numpy()  # [256]
ols_bias = ols.predictor_intercept.item()

print("\n" + "=" * 100)
print("WEIGHTS COMPARISON (256-dimensional weight vector)")
print("=" * 100)

print(f"\n{'Method':<12} {'Mean':<20} {'Std':<20} {'Min':<20} {'Max':<20} {'L2 Norm':<20}")
print("-" * 100)
print(f"{'SGD_1000':<12} {np.mean(sgd_1000_weights):<20.10f} {np.std(sgd_1000_weights):<20.10f} {np.min(sgd_1000_weights):<20.10f} {np.max(sgd_1000_weights):<20.10f} {np.linalg.norm(sgd_1000_weights):<20.10f}")
print(f"{'SGD_2000':<12} {np.mean(sgd_2000_weights):<20.10f} {np.std(sgd_2000_weights):<20.10f} {np.min(sgd_2000_weights):<20.10f} {np.max(sgd_2000_weights):<20.10f} {np.linalg.norm(sgd_2000_weights):<20.10f}")
print(f"{'OLS':<12} {np.mean(ols_weights):<20.10f} {np.std(ols_weights):<20.10f} {np.min(ols_weights):<20.10f} {np.max(ols_weights):<20.10f} {np.linalg.norm(ols_weights):<20.10f}")

print("\n" + "=" * 100)
print("BIAS COMPARISON")
print("=" * 100)
print(f"\n{'Method':<12} {'Bias Value':<20}")
print("-" * 100)
print(f"{'SGD_1000':<12} {sgd_1000_bias:<20.10f}")
print(f"{'SGD_2000':<12} {sgd_2000_bias:<20.10f}")
print(f"{'OLS':<12} {ols_bias:<20.10f}")

print("\n" + "=" * 100)
print("DIFFERENCES BETWEEN METHODS")
print("=" * 100)

# Weight differences
diff_1000_2000 = sgd_2000_weights - sgd_1000_weights
diff_1000_ols = ols_weights - sgd_1000_weights
diff_2000_ols = ols_weights - sgd_2000_weights

print(f"\nWeight Differences (L2 norm):")
print(f"  SGD_2000 - SGD_1000: {np.linalg.norm(diff_1000_2000):.10f}")
print(f"  OLS - SGD_1000:      {np.linalg.norm(diff_1000_ols):.10f}")
print(f"  OLS - SGD_2000:      {np.linalg.norm(diff_2000_ols):.10f}")

print(f"\nWeight Differences (Mean absolute):")
print(f"  SGD_2000 - SGD_1000: {np.mean(np.abs(diff_1000_2000)):.10f}")
print(f"  OLS - SGD_1000:      {np.mean(np.abs(diff_1000_ols)):.10f}")
print(f"  OLS - SGD_2000:      {np.mean(np.abs(diff_2000_ols)):.10f}")

print(f"\nWeight Differences (Max absolute):")
print(f"  SGD_2000 - SGD_1000: {np.max(np.abs(diff_1000_2000)):.10f}")
print(f"  OLS - SGD_1000:      {np.max(np.abs(diff_1000_ols)):.10f}")
print(f"  OLS - SGD_2000:      {np.max(np.abs(diff_2000_ols)):.10f}")

# Bias differences
print(f"\nBias Differences:")
print(f"  SGD_2000 - SGD_1000: {sgd_2000_bias - sgd_1000_bias:.10f}")
print(f"  OLS - SGD_1000:      {ols_bias - sgd_1000_bias:.10f}")
print(f"  OLS - SGD_2000:      {ols_bias - sgd_2000_bias:.10f}")

print("\n" + "=" * 100)
print("WEIGHT VECTOR SIMILARITY (Cosine Similarity)")
print("=" * 100)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sim_1000_2000 = cosine_similarity(sgd_1000_weights, sgd_2000_weights)
cos_sim_1000_ols = cosine_similarity(sgd_1000_weights, ols_weights)
cos_sim_2000_ols = cosine_similarity(sgd_2000_weights, ols_weights)

print(f"\nCosine Similarity (1.0 = identical, -1.0 = opposite):")
print(f"  SGD_1000 vs SGD_2000: {cos_sim_1000_2000:.10f}")
print(f"  SGD_1000 vs OLS:      {cos_sim_1000_ols:.10f}")
print(f"  SGD_2000 vs OLS:      {cos_sim_2000_ols:.10f}")

print("\n" + "=" * 100)
print("WEIGHT DISTRIBUTION STATISTICS")
print("=" * 100)

print(f"\n{'Method':<12} {'Percentiles':<60}")
print("-" * 100)
for method_name, weights in [('SGD_1000', sgd_1000_weights), ('SGD_2000', sgd_2000_weights), ('OLS', ols_weights)]:
    percentiles = np.percentile(weights, [0, 25, 50, 75, 100])
    print(f"{method_name:<12} P0={percentiles[0]:>10.6f} P25={percentiles[1]:>10.6f} P50={percentiles[2]:>10.6f} P75={percentiles[3]:>10.6f} P100={percentiles[4]:>10.6f}")

print("\n" + "=" * 100)
print("LARGEST DIFFERENCES (Top 10 weight indices)")
print("=" * 100)

# Find indices with largest differences
diff_2000_ols_abs = np.abs(diff_2000_ols)
top_10_indices = np.argsort(diff_2000_ols_abs)[-10:][::-1]

print(f"\nTop 10 indices with largest |OLS - SGD_2000| differences:")
print(f"{'Index':<10} {'SGD_2000':<20} {'OLS':<20} {'Difference':<20}")
print("-" * 100)
for idx in top_10_indices:
    print(f"{idx:<10} {sgd_2000_weights[idx]:<20.10f} {ols_weights[idx]:<20.10f} {diff_2000_ols[idx]:<20.10f}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

print("\n1. WEIGHT MAGNITUDE:")
print(f"   - SGD_1000 L2 norm: {np.linalg.norm(sgd_1000_weights):.6f}")
print(f"   - SGD_2000 L2 norm: {np.linalg.norm(sgd_2000_weights):.6f}")
print(f"   - OLS L2 norm:      {np.linalg.norm(ols_weights):.6f}")
print(f"   → OLS has {'larger' if np.linalg.norm(ols_weights) > np.linalg.norm(sgd_2000_weights) else 'smaller'} weights than SGD_2000")

print("\n2. WEIGHT DIFFERENCES:")
print(f"   - SGD_2000 differs from SGD_1000 by L2 = {np.linalg.norm(diff_1000_2000):.6f}")
print(f"   - OLS differs from SGD_2000 by L2 = {np.linalg.norm(diff_2000_ols):.6f}")
print(f"   → OLS and SGD_2000 are {'more similar' if np.linalg.norm(diff_2000_ols) < np.linalg.norm(diff_1000_2000) else 'more different'} than SGD_1000 and SGD_2000")

print("\n3. DIRECTIONAL SIMILARITY:")
print(f"   - Cosine similarity SGD_2000 vs OLS: {cos_sim_2000_ols:.6f}")
print(f"   → {'Very similar' if cos_sim_2000_ols > 0.99 else 'Similar' if cos_sim_2000_ols > 0.9 else 'Different'} directions")

print("\n4. BIAS DIFFERENCES:")
print(f"   - Bias difference OLS - SGD_2000: {ols_bias - sgd_2000_bias:.10f}")
print(f"   → Bias values are {'very close' if abs(ols_bias - sgd_2000_bias) < 0.0001 else 'different'}")

print("=" * 100)

