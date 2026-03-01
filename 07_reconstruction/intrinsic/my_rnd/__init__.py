"""
My RND module: flat-obs encoder + predictor/target, no update_proportion/kappa/reward norm.
All learning data comes from replay buffer sample(); direct batch shape (batch_size, obs_dim).
"""
from .rnd import MyRND, ObservationEncoder
