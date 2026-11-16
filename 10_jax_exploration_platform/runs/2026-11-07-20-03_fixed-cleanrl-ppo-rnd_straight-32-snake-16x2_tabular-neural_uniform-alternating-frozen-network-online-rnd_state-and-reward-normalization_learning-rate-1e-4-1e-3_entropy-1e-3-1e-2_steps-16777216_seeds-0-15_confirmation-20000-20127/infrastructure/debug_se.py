import sys; import numpy as np
sys.path.insert(0, '/p/rlprojects/RND/10_jax_exploration_platform/scripts/grid_initialization_campaign')
import fixed_settings_report as R
v = np.random.default_rng(3).exponential(1000.0, 128)
print(R.grouped_standard_errors(v, list(range(128)), 10, 7), v.std(ddof=1)/np.sqrt(128))
