#!/usr/bin/env python
"""Collect the fixed `atari_frames` evaluation domain: MontezumaRevenge frames from
fixed-seed random play, preprocessed the way RND's predictor sees them (grayscale, 84x84,
single frame), frozen as one .npy artifact.

MontezumaRevenge is the flagship RND domain. Random play only reaches the first room, which
is fine for a convergence-rate domain: the frames are distinct high-dimensional observations
(the ale's own frame variation: sprites, timers, the character's pose) whose per-frame visit
counts the harness controls exactly.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python collect_atari_frames.py
Writes ../../domains_data/atari_frames.npy (P, 84, 84) float32 in [0, 1] + .meta.json.
"""
import json
import os

import ale_py  # noqa: F401  (registers ALE envs)
import cv2
import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "domains_data")

SEED = 0
N_FRAMES = 512
COLLECT_EVERY = 12       # keep every 12th emitted frame so consecutive kept frames differ
GAME = "ALE/MontezumaRevenge-v5"


def preprocess(frame: np.ndarray) -> np.ndarray:
    """RND-style single-frame preprocessing: grayscale, 84x84, [0, 1] float32.
    before: (210, 160, 3) uint8; after: (84, 84) float32 in [0, 1]"""
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA)
    return (g.astype(np.float32) / 255.0)


def main():
    """Random play until N_FRAMES kept frames; freeze."""
    env = gym.make(GAME, frameskip=4)
    frames, t, ep = [], 0, 0
    obs, _ = env.reset(seed=SEED)
    while len(frames) < N_FRAMES:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        t += 1
        if t % COLLECT_EVERY == 0:
            frames.append(preprocess(obs))
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=SEED + ep)
    env.close()
    arr = np.stack(frames)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "atari_frames.npy"), arr)
    meta = {"seed": SEED, "game": GAME, "n_frames": int(arr.shape[0]),
            "collect_every": COLLECT_EVERY, "episodes_used": ep + 1,
            "shape": list(arr.shape), "preprocessing": "grayscale, 84x84 INTER_AREA, /255"}
    with open(os.path.join(OUT_DIR, "atari_frames.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
