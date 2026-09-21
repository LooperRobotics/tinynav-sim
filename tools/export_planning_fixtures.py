"""Export golden fixtures from the Python reference for C++ alignment tests.

Run inside the tinynav container (see AGENTS.md):

    docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \\
      --entrypoint /bin/bash uniflexai/tinynav:latest -c \\
      'source /opt/ros/humble/setup.bash && \\
       PYTHONPATH=/ws/reference python3 tools/export_planning_fixtures.py'

Writes fixtures/planning/*.npy (gitignored). Every fixture is a set of
<numbered>.npy files per case: inputs and the reference outputs produced by
the UNMODIFIED reference functions (tinynav.core.planning_node / scipy).
The C++ test replays the same calls through the port and compares.

The DWA cost block is a verbatim copy of planning_node.py::sync_callback's
cost_function closure (the closure is not importable); the copy is marked and
line-referenced, and its inputs (scores/path_costs/...) are themselves checked
against score_trajectories_by_ESDF fixtures, so the only freedom the copy has
is transcription error, which is exactly what the comparison catches.
"""

import os
import sys

sys.path.insert(0, "/ws/reference")

import numpy as np
from scipy.ndimage import distance_transform_edt  # noqa: E402  (reference's own call)

from tinynav.core import planning_node as pn  # noqa: E402
from tinynav.core.robot_specs import ObstacleConfig, ROBOT_CONFIG  # noqa: E402

OUT = "/ws/fixtures/planning"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260918)


def save(name, arr):
    np.save(f"{OUT}/{name}.npy", np.ascontiguousarray(arr))
    print(f"  {name}: {np.shape(arr)}")


def pose(seed):
    r = np.random.default_rng(seed)
    T = np.eye(4)
    a, b, c = r.uniform(-0.2, 0.2, 3)
    T[:3, :3] = pn.quat_to_matrix(
        __import__("scipy.spatial.transform", fromlist=["Rotation"])
        .Rotation.from_euler("xyz", [a, b, c])
        .as_quat()
    )
    T[:3, 3] = r.uniform(-1.0, 1.0, 3)
    return T


print("== 1. run_raycasting_loopy ==")
for case, (h, w, step) in {"a": (48, 64, 4), "b": (30, 40, 3), "c": (48, 64, 4)}.items():
    depth = rng.uniform(0.2, 3.0, (h, w)).astype(np.float32)
    if case != "b":  # b keeps every pixel valid
        depth.flat[rng.integers(0, h * w, 20)] = 0.0
        depth.flat[rng.integers(0, h * w, 10)] = np.float32("nan")
    T = pose(hash(case) % 1000)
    grid_shape = (30, 30, 10)
    origin = np.array([-1.5, -1.5, -0.5])
    args = (depth, T, grid_shape, 40.0, 41.0, 20.0, 15.0, origin, step, 0.1)
    out = pn.run_raycasting_loopy(*args)
    save(f"raycast_{case}_depth", depth)
    save(f"raycast_{case}_T", T)
    save(f"raycast_{case}_origin", origin)
    save(f"raycast_{case}_cfg", np.array([40.0, 41.0, 20.0, 15.0, step, 0.1]))
    save(f"raycast_{case}_out", out)

print("== 2. generate_trajectory_library_3d ==")
for case, kwargs in {
    "default": {},
    "vxmin": dict(max_linear_vel=0.6, max_angular_vel=0.75, max_path_len_m=2.5,
                  max_lat_acc=0.5, min_linear_vel=0.2),
    "capped": dict(max_linear_vel=0.5, max_angular_vel=np.pi / 3, max_path_len_m=0.2,
                   max_lat_acc=0.05, min_linear_vel=0.0),
}.items():
    trajs, params = pn.generate_trajectory_library_3d(**kwargs)
    save(f"traj_{case}_out", trajs)
    save(f"traj_{case}_params", params)

trajs, params = pn.generate_predefined_trajectory_vocabularies()
save("vocab_out", trajs)
save("vocab_params", params)

print("== 3. build_obstacle_map ==")
grid = rng.uniform(-0.1, 0.1, (40, 40, 12))
grid[5:25, 8:12, 2:8] = 0.08   # a wall
grid[30, 30, 3] = 0.09         # single-voxel noise
grid[10, 30, 0:6] = 0.07       # floating blob
origin3 = np.array([-1.0, -1.0, -0.3])
for case, (config, min_span) in {
    "strict": (ObstacleConfig(), None),
    "dilate": (ObstacleConfig(dilation_cells=2), None),
    "relax": (ObstacleConfig(), np.where(
        (np.arange(1600).reshape(40, 40) % 7 == 0), 0.2, 0.05).astype(np.float32)),
}.items():
    mask = pn.build_obstacle_map(grid, origin3, 0.1, robot_z=-0.1,
                                 config=config, min_span_map=min_span)
    save(f"obstacle_{case}_grid", grid)
    save(f"obstacle_{case}_origin", origin3)
    save(f"obstacle_{case}_robot_z", np.array(-0.1))
    if min_span is not None:
        save(f"obstacle_{case}_minspan", min_span)
    save(f"obstacle_{case}_out", mask.astype(np.uint8))

print("== 4. roll_occupancy_grid ==")
grid = rng.uniform(-0.2, 0.2, (20, 24, 6))
old_origin = np.array([-1.0, -1.2, -0.3])
for case, new_origin in {
    "x": np.array([-0.85, -1.2, -0.3]),
    "xyz": np.array([-0.75, -0.95, -0.05]),
    "back": np.array([-1.35, -1.2, -0.3]),
    "identity": np.array([-0.96, -1.2, -0.3]),
}.items():
    rolled, updated = pn.roll_occupancy_grid(grid, old_origin, new_origin, 0.1)
    save(f"roll_{case}_grid", grid)
    save(f"roll_{case}_old", old_origin)
    save(f"roll_{case}_new", new_origin)
    save(f"roll_{case}_out", rolled)
    save(f"roll_{case}_out_origin", updated)

print("== 5. footprint_lattice ==")
for name, cfg in {"go2": ROBOT_CONFIG, "b2": None, "g1": None}.items():
    from tinynav.core import robot_specs as rs
    cfg = {"go2": rs.GO2_CONFIG, "b2": rs.B2_CONFIG, "g1": rs.G1_CONFIG}[name]
    fl, rl, hw = cfg.footprint_from_control()
    fwd, lat = pn.footprint_lattice(fl, rl, hw, cfg.safety_radius)
    save(f"lattice_{name}_cfg", np.array([fl, rl, hw, cfg.safety_radius]))
    save(f"lattice_{name}_fwd", fwd)
    save(f"lattice_{name}_lat", lat)

print("== 6. score_trajectories_by_ESDF ==")
lib, lib_params = pn.generate_trajectory_library_3d(
    max_linear_vel=0.6, max_angular_vel=0.75, max_path_len_m=2.5,
    max_lat_acc=0.5, min_linear_vel=0.2)
vocab, vocab_params = pn.generate_predefined_trajectory_vocabularies()
trajs = np.concatenate([lib[:30], vocab], axis=0)
params = np.concatenate([lib_params[:30], vocab_params], axis=0)
# one fully off-grid trajectory appended
off = trajs[-1:].copy()
off[..., 0] += 100.0
trajs = np.concatenate([trajs, off], axis=0)
params = np.concatenate([params, [[-0.3, 0.0]]], axis=0).astype(np.float64)

esdf = rng.uniform(0.0, 1.0, (80, 80)).astype(np.float32)
esdf[30:45, 10:20] = 0.0
esdf[:, :3] = 0.0
path_dist = rng.uniform(0.0, 3.0, (80, 80)).astype(np.float32)
remaining = rng.uniform(0.0, 20.0, (80, 80)).astype(np.float32)
heading = rng.uniform(-np.pi, np.pi, (80, 80)).astype(np.float32)
origin2 = np.array([-2.0, -2.0])
fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
scores, occ, pcosts, erem, ehead = pn.score_trajectories_by_ESDF(
    np.ascontiguousarray(trajs), esdf, path_dist, remaining, heading,
    origin2, 0.05, ROBOT_CONFIG.safety_radius, fl, rl, hw)
save("score_trajs", trajs)
save("score_esdf", esdf)
save("score_path_dist", path_dist)
save("score_remaining", remaining)
save("score_heading", heading)
save("score_origin", origin2)
save("score_cfg", np.array([ROBOT_CONFIG.safety_radius, fl, rl, hw, 0.05]))
save("score_out_scores", np.array(scores))
save("score_out_occ", np.array(occ, dtype=np.int64))
save("score_out_path_costs", np.array(pcosts))
save("score_out_end_remainings", np.array(erem))
save("score_out_end_heading_errs", np.array(ehead))

print("== 7. build_route_fields ==")
for case, route in {
    "straight": np.array([[0.1, 1.5], [3.9, 1.5]]),
    "elbow": np.array([[0.2, 0.3], [0.2, 2.4], [2.8, 2.4], [2.8, 3.9]]),
    "zigzag": np.array([[0.1, 0.1], [1.2, 0.9], [2.1, 0.2], [3.3, 1.1], [3.9, 0.6]]),
    "offgrid": np.array([[50.0, 50.0], [52.0, 50.0]]),
    "degenerate": np.array([[1.0, 1.0], [1.0, 1.0]]),
}.items():
    pd, rem, rh, has = pn.build_route_fields(
        route.astype(float), (80, 70), np.array([-2.0, -2.0]), 0.05)
    save(f"route_{case}_xy", route)
    save(f"route_{case}_path_dist", pd)
    save(f"route_{case}_remaining", rem)
    save(f"route_{case}_heading", rh)
    save(f"route_{case}_has", np.array(1 if has else 0, dtype=np.int64))

print("== 8. scipy EDT (reference semantics) ==")
mask = rng.uniform(0, 1, (50, 60)) > 0.85
dist, inds = distance_transform_edt(~mask, return_indices=True)
save("edt_random_mask", mask.astype(np.uint8))
save("edt_random_dist", dist)
save("edt_random_inds", inds.astype(np.int64))  # (2, 50, 60); indices are tie-checked, not compared
tie = np.zeros((9, 9), dtype=bool)
tie[4, 0] = True
tie[4, 8] = True  # symmetric: indices may tie-break differently
tie_dist = distance_transform_edt(~tie)
save("edt_tie_mask", tie.astype(np.uint8))
save("edt_tie_dist", tie_dist)

print("== 9. scalars: reverse_armed / speed_from_clearance ==")
cases = [(0.3, 5, 0.05), (0.32, 5, 0.05), (0.36, 5, 0.05), (10.0, 0, 0.05), (10.0, 3, 0.05)]
rev = [pn.reverse_armed(*c) for c in cases]
save("reverse_cases", np.array(cases))
save("reverse_out", np.array(rev, dtype=np.int64))
speed_cases = [(1.0, 0.5, 0.8), (0.2, 0.5, 0.8), (10.0, 0.5, 0.8), (0.0, 1.0, 0.8)]
# planning_node._speed_from_clearance is a node method (not importable); its
# body is a single np.interp over [clear_c0_m, clear_open_m] with a latency
# discount — construct the golden straight from np.interp.
speeds = [float(np.interp(max(0.0, c - v * 0.2), [0.35, 1.0], [0.2, s]))
          for c, v, s in speed_cases]
save("speed_cases", np.array(speed_cases))
save("speed_out", np.array(speeds))

print("== 10. DWA cost + selection (verbatim closure copy, planning_node.py:1308) ==")


def reference_cost(i, trajectories, params, scores, path_costs, end_remainings,
                   end_heading_errs, has_route, target, last_param, should_reverse,
                   w_clearance=200.0, w_route_progress=100.0, w_path_follow=80.0,
                   w_goal_terminal=100.0, route_terminal_band=0.5, w_route_heading=60.0):
    # VERBATIM copy of PlanningNode.sync_callback's cost_function closure
    # (planning_node.py:1308-1354); `self.` fields became the keyword args above.
    def _end_heading_error(pose7, goal):
        return pn.angle_between(
            np.arctan2(goal[1] - pose7[1], goal[0] - pose7[0]),
            pn.heading_of_pose7(pose7))

    traj, param = trajectories[i], params[i]
    reverse_gate_penalty = 0.0 if (param[0] < 0.0) == should_reverse else 1e9
    traj_end = np.array(traj[-1, :3])
    target_end = target if target is not None else traj_end
    dist = np.linalg.norm(traj_end - target_end)
    smooth = abs(last_param[0] - param[0]) + abs(last_param[1] - param[1])
    heading_penalty = 0.0
    if dist > 0.3:
        to_goal = w_route_heading * _end_heading_error(traj[-1], target_end)
        if has_route:
            fade = pn.route_band_fade(end_remainings[i], route_terminal_band)
            heading_penalty = (
                pn.route_heading_penalty(w_route_heading, end_heading_errs[i],
                                         end_remainings[i], route_terminal_band)
                + (1.0 - fade) * to_goal)
        else:
            heading_penalty = to_goal
    if not has_route:
        positional = 100 * dist
    else:
        terminal = 0.0
        if target is not None:
            terminal = (w_goal_terminal
                        * (1.0 - pn.route_band_fade(end_remainings[i], route_terminal_band))
                        * float(np.linalg.norm(traj[-1, :2] - target[:2])))
        positional = (w_route_progress * end_remainings[i]
                      + w_path_follow * path_costs[i]
                      + terminal)
    return (scores[i] * w_clearance
            + positional
            + 10 * smooth
            + heading_penalty
            + reverse_gate_penalty)


lib, lib_params = pn.generate_trajectory_library_3d(
    max_linear_vel=0.6, max_angular_vel=0.75, max_path_len_m=2.5,
    max_lat_acc=0.5, min_linear_vel=0.2)
vocab, vocab_params = pn.generate_predefined_trajectory_vocabularies()
trajs = np.concatenate([lib, vocab], axis=0)
params = np.concatenate([lib_params, vocab_params], axis=0)
scores = rng.uniform(0.0, 2.0, len(trajs))
scores[rng.integers(0, len(trajs), 8)] = np.inf
path_costs = rng.uniform(0.0, 3.0, len(trajs))
end_remainings = rng.uniform(0.0, 25.0, len(trajs))
end_heading_errs = rng.uniform(0.0, np.pi, len(trajs))
for case, (has_route, target, last_param, should_reverse) in {
    "route": (True, np.array([3.0, 1.0, 0.1]), np.array([0.3, 0.1]), False),
    "noroute": (False, np.array([3.0, 1.0, 0.1]), np.array([0.3, 0.1]), False),
    "reverse": (True, np.array([3.0, 1.0, 0.1]), np.array([-0.3, 0.0]), True),
}.items():
    costs = np.array([reference_cost(i, trajs, params, scores, path_costs,
                                     end_remainings, end_heading_errs, has_route,
                                     target, last_param, should_reverse)
                      for i in range(len(trajs))])
    save(f"cost_{case}_trajs", trajs)
    save(f"cost_{case}_params", params)
    save(f"cost_{case}_scores", scores)
    save(f"cost_{case}_path_costs", path_costs)
    save(f"cost_{case}_end_remainings", end_remainings)
    save(f"cost_{case}_end_heading_errs", end_heading_errs)
    save(f"cost_{case}_cfg", np.array([int(has_route), *(target if target is not None else [0, 0, 0]),
                                       *last_param, int(should_reverse)]))
    save(f"cost_{case}_out", costs)
    save(f"cost_{case}_argmin", np.array(int(np.argmin(costs)), dtype=np.int64))

print(f"done -> {OUT}")
