"""Which wall boxes MJX tests against the ant when the number of geometry pairs is limited.

MJX keeps, per kind of geometry pair, the `max_geom_pairs` pairs whose bounding spheres overlap
most: it ranks every pair by (distance between the two centres) minus (the two bounding radii)
and keeps the smallest (`mujoco/mjx/_src/collision_driver.py`, `collision`). A wall box that is
a long run of cells has a large bounding sphere, so it ranks ahead of a short box that the ant
actually touches. This script counts, for each map and each limit, the places where a body of
the ant that touches a wall is not tested against that wall.

For every wall box and each of its four vertical faces, points are placed every 0.25 m along
the face, on the open side, at the height 0.5 m:

  torso          the torso sphere (radius 0.25 m) touching the face. Tested pairs: torso against
                 the `max_geom_pairs` best-ranked boxes.
  one leg part   a leg capsule's centre 0.08 m from the face (its radius), lying along the face.
                 Tested pairs: the `max_geom_pairs` best-ranked of all (capsule, box) pairs of the
                 ant, with the other 11 capsules placed around a torso 0.6 m further from the
                 face, where the ant's own model puts them at its spawn pose.

No physics is stepped: the ranking is arithmetic on the model's positions and radii. Runs inside
a Slurm job through the test runner, which calls a function named `main`:
  bash <run folder>/infrastructure/submit_tests.sh -o "python_functions=test_* main" -s <this file>
It writes wall_pairs_kept_on_processors.json beside this folder and prints a Markdown table.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FOLDER = Path(__file__).resolve().parent.parent
PLATFORM = FOLDER.parents[3]
SPACING = 0.25          # metres between sample points along a wall face
HEIGHT = 0.5            # metres, the height of the sampled bodies
LIMITS = (4, 8, 16)     # values of max_geom_pairs


def face_points(box, grid, scaling):
    """Points on the open side of the four vertical faces of one wall box.

    before: box = (x, y, half_x, half_y) in metres; grid = the maze map (1 = wall)
    after:  rows (x, y, normal_x, normal_y): a point on a face whose outward side is an open cell
    """
    import numpy as np
    rows, cols = len(grid), len(grid[0])
    x, y, hx, hy = box
    points = []
    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        # GOAL: walk along this face; keep the points whose neighbour cell across the face is open.
        length = hy if nx else hx
        for offset in np.arange(-length + SPACING / 2, length, SPACING):
            px = x + nx * hx + (0.0 if nx else offset)
            py = y + ny * hy + (offset if nx else 0.0)
            j = int(np.floor((px + nx * 0.5) / scaling + cols / 2.0))
            i = int(np.floor(rows / 2.0 - (py + ny * 0.5) / scaling))
            if 0 <= i < rows and 0 <= j < cols and grid[i][j] == 0:
                points.append((px, py, nx, ny))
    return points


def measure_map(map_name):
    """Per limit: at how many face points the touched wall is not among the tested pairs."""
    import mujoco
    import numpy as np
    from exploration_platform.envs.antmaze.am_common import (SCALING, build_antmaze_xml, merged_wall_boxes,
                                                             preset)
    from exploration_platform.envs.pointmaze.pm_common import MAPS
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(preset(map_name)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    boxes = [g for g, name in enumerate(names) if name.startswith("block_")]
    capsules = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
    torso = names.index("torso_geom")
    box_pos, box_radius = model.geom_pos[boxes], model.geom_rbound[boxes]
    # the capsules' centres relative to the torso at the spawn pose, and their bounding radii
    capsule_offsets = data.geom_xpos[capsules] - data.geom_xpos[torso]
    capsule_radius = model.geom_rbound[capsules]
    walls = merged_wall_boxes(map_name)
    assert len(walls) == len(boxes)
    counts = {k: {"torso": 0, "one leg part": 0} for k in LIMITS}
    n_points = 0
    for b, wall in enumerate(walls):
        for px, py, nx, ny in face_points(wall, MAPS[map_name], SCALING):
            n_points += 1
            # GOAL: the torso touching the face: rank the boxes as MJX does, see whether box b is kept.
            centre = np.array([px + nx * 0.25, py + ny * 0.25, HEIGHT])
            score = np.linalg.norm(box_pos - centre, axis=1) - (box_radius + model.geom_rbound[torso])
            order = np.argsort(score, kind="stable")
            # GOAL: one capsule touching the face, the others around a torso 0.6 m further out.
            body = np.array([px + nx * 0.68, py + ny * 0.68, HEIGHT])
            centres = body + capsule_offsets
            centres[0] = np.array([px + nx * 0.08, py + ny * 0.08, HEIGHT])
            pair_score = (np.linalg.norm(centres[:, None, :] - box_pos[None, :, :], axis=2)
                          - (capsule_radius[:, None] + box_radius[None, :]))
            pair_order = np.argsort(pair_score.reshape(-1), kind="stable")
            for k in LIMITS:
                counts[k]["torso"] += int(b not in order[:k])
                counts[k]["one leg part"] += int(b not in pair_order[:k])   # capsule 0 is row 0: its pair index is b
    return {"map": map_name, "wall_boxes": len(boxes), "face_points": n_points,
            "leg_pairs": len(capsules) * len(boxes), "not_tested": counts}


def main():
    """Measure the three maps, write the record and print the table."""
    sys.path.insert(0, str(PLATFORM / "src"))
    rows = [measure_map(name) for name in ("umaze", "medium", "large")]
    record = {"measured_pacific": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT"),
              "spacing_m": SPACING, "height_m": HEIGHT, "maps": rows}
    (FOLDER / "wall_pairs_kept_on_processors.json").write_text(json.dumps(record, indent=1))
    print("| map | wall boxes | face points | body | " + " | ".join(
        f"not tested at<br>geometry pairs {k}" for k in LIMITS) + " |")
    print("|---|---:|---:|---|" + "---:|" * len(LIMITS))
    for row in rows:
        for body in ("torso", "one leg part"):
            print(f"| {row['map']} | {row['wall_boxes']} | {row['face_points']} | {body} | " + " | ".join(
                f"{row['not_tested'][k][body]} ({100 * row['not_tested'][k][body] / row['face_points']:.0f}%)"
                for k in LIMITS) + " |")


if __name__ == "__main__":
    main()
