"""
Minimal patch showing how to fix 04_many_exploration_method.py.

This file shows the exact lines to change. Do NOT run this file directly.
Apply these changes to 07_reconstruction/04_many_exploration_method.py.
"""

# ============================================================
# FIX 1: Remove TerminateOnTimeLimitWrapper (CRITICAL)
# ============================================================
#
# In 04_many_exploration_method.py, REMOVE these two lines:
#
# BEFORE (broken, line ~131 and ~207):
#   base_env = TerminateOnTimeLimitWrapper(base_env)
#   eval_base = TerminateOnTimeLimitWrapper(eval_base)
#
# AFTER (fixed):
#   (just delete both lines)
#
# Also remove TerminateOnTimeLimitWrapper from the import:
#
# BEFORE:
#   from env_wrapper.point_maze_wrappers import (
#       FixedGoalWrapper,
#       FixedStartWrapper,
#       RemoveGoalWrapper,
#       TerminateOnTimeLimitWrapper,
#       PositionVisitCountWrapper,
#       PositionVelocityVisitCountWrapper,
#       ComputeIntrinsicRewardWrapper,
#   )
#
# AFTER:
#   from env_wrapper.point_maze_wrappers import (
#       FixedGoalWrapper,
#       FixedStartWrapper,
#       RemoveGoalWrapper,
#       PositionVisitCountWrapper,
#       PositionVelocityVisitCountWrapper,
#       ComputeIntrinsicRewardWrapper,
#   )

# ============================================================
# FIX 2 (Optional): Restore DataLoader mini-batching in RND.update()
# ============================================================
#
# In intrinsic/intrinsic_method/rnd.py, replace the update() method:
#
# BEFORE (current broken version):
#
#     def update(self, samples):
#         x = self._get_feature_tensor(samples)
#         if self.use_obs_norm and self.obs_rms is not None:
#             self.obs_rms.update(x.detach().cpu().numpy())
#         x = self._normalize_obs(x)
#         self.opt.zero_grad()
#         if self.linear_rnd:
#             tgt = self.target.forward_target(x).detach()
#             pred = self.target.forward_predictor(x)
#             loss = (pred - tgt).pow(2).mean()
#         else:
#             src = self.predictor(x)
#             if self.n_predictors == 1:
#                 src = src.unsqueeze(1)
#             with torch.no_grad():
#                 tgt = self.target(x)
#             loss = self._dist_ensemble(src, tgt).mean()
#         loss.backward()
#         self.opt.step()
#
# AFTER (with DataLoader mini-batching, matches old MyRND behavior):
#
#     def update(self, samples):
#         from torch.utils.data import DataLoader, TensorDataset
#         x = self._get_feature_tensor(samples)
#         if self.use_obs_norm and self.obs_rms is not None:
#             self.obs_rms.update(x.detach().cpu().numpy())
#         x = self._normalize_obs(x)
#         dataset = TensorDataset(x)
#         loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
#         for batch in loader:
#             o = batch[0]
#             self.opt.zero_grad()
#             if self.linear_rnd:
#                 tgt = self.target.forward_target(o).detach()
#                 pred = self.target.forward_predictor(o)
#                 loss = (pred - tgt).pow(2).mean()
#             else:
#                 src = self.predictor(o)
#                 if self.n_predictors == 1:
#                     src = src.unsqueeze(1)
#                 with torch.no_grad():
#                     tgt = self.target(o)
#                 loss = self._dist_ensemble(src, tgt).mean()
#             loss.backward()
#             self.opt.step()
