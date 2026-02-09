# Code study: [Title]

**Date:** YYYY-MM-DD  
**Source:** path or repo (e.g. `07_reconstruction/new_rl.py`)

---

## What it does

- One-line purpose:
- In context of the project:

---

## Entry point / flow

- Where does execution start?
- Main call path (high level):

---

## Key concepts

| Concept | Where | Notes |
|--------|--------|--------|
|        |        |        |

---

## Code references

### Snippet 1: [description]

```python
# path/to/file.py (lines X–Y)
# paste or describe
```

Notes:

---

### Snippet 2: [description]

```python
# path/to/file.py (lines X–Y)
```

Notes:

---

## Dependencies / imports

- Important modules or functions used:
- External libs that matter:

---

## Questions / TODOs

- [ ] 
- [ ] 

---

## Summary

(1–3 sentences: what you learned or how this fits together.)

---

## Scratchpad: derived training sizes

Assumptions (edit as needed):
- `num_steps = 128`
- `num_minibatches = 4`

```python
# Given
total_timesteps: int = 500_000      # total timesteps of the experiment
num_envs: int = 4                   # number of parallel environments
num_steps: int = 128                # steps per env per rollout
num_minibatches: int = 4            # mini-batches per update

# Derived
batch_size: int = num_envs * num_steps                 # per-update batch: 4 * 128 = 512
minibatch_size: int = batch_size // num_minibatches    # per-SGD step: 512 // 4 = 128
num_iterations: int = total_timesteps // batch_size    # total updates: 500_000 // 512 = 976
```