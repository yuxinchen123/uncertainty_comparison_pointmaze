"""Lives, step counts and resets of every environment in a saved learning-check state."""
import os, pickle, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
payload = pickle.load(open(sys.argv[1], "rb"))
env = payload["state"].environment
game = env.game
print("fields of the environment state:", [f for f in env._fields] if hasattr(env, "_fields") else type(env))
lives = np.asarray(game.lives); steps = np.asarray(env.step_count); resets = np.asarray(env.reset_count)
print("lives distribution:", {int(v): int(c) for v, c in zip(*np.unique(lives, return_counts=True))})
print("step_count min/median/max:", int(steps.min()), int(np.median(steps)), int(steps.max()))
print("reset_count min/median/max:", int(resets.min()), int(np.median(resets)), int(resets.max()))
print("iteration:", int(payload["state"].iteration), "room ids:", {int(v): int(c) for v, c in zip(*np.unique(np.asarray(game.room_id), return_counts=True))})
