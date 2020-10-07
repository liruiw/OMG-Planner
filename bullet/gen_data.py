# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys

import cv2
import scipy.io as sio
from .panda_scene import PandaYCBEnv
import numpy as np
import IPython
from omg.core import *
from omg.util import *
from omg.config import cfg
import time

np.set_printoptions(2)


def record_traj(
    file_name, observations, trajectory, pose, path, proj_matrix, view_matrix, target_name, goals, write_images
):
    
    scene_mat = {}
    scene_mat["path"] = path
    scene_mat["pose"] = pose
    scene_mat["trajectory"] = trajectory
    scene_mat["states"] = np.array([obs[1] for obs in observations])
    scene_mat["proj_matrix"] = env._proj_matrix
    scene_mat["view_matrix"] = env._view_matrix
    scene_mat["goals"] = goals
    scene_mat["target_name"] = target_name
    sio.savemat(file_name + ".mat", scene_mat)
    print("save path:", file_name)
    img = [obs[0] for obs in observations]
    if write_images:
        for i in range(len(img)):
            cv2.imwrite("{}_color_{}.png".format(file_name, i), img[i][:, :, [2, 1, 0]])
            cv2.imwrite(
                "{}_depth_{}.png".format(file_name, i),
                (img[i][..., [3]] * 5000).astype(np.uint16),
            )
            cv2.imwrite("{}_mask_{}.png".format(file_name, i), img[i][..., [4]])


def check_scene(poses):
    drop_on_ground = [p[2] < -0.1 for p in poses[:-2]]
    return np.sum(drop_on_ground) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--save_dir", help="filename", type=str, default="data/demonstrations/"
    )
    parser.add_argument("-v", "--vis", help="visualize", action="store_true")
    parser.add_argument("-pv", "--planner_vis", help="visualize", action="store_true")    
    parser.add_argument("-n", "--num", help="number of files", type=int, default=10)
    parser.add_argument("-w", "--write_images", help="write images", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")    

    args = parser.parse_args()
    env = PandaYCBEnv(renders=args.vis, egl_render=args.egl)
    env.reset()

    # planner setup and cache
    cfg.smoothness_base_weight = 0.1    
    cfg.dynamic_timestep = False
    cfg.collision_point_num = 15
    cfg.report_cost = False
    cfg.traj_init = "grasp"
    cfg.scene_file = ""
    cfg.reach_tail_length = 3
    cfg.standoff_dist = 0.05
    cfg.goal_set_max_num = 100
    cfg.timesteps = 50
    cfg.get_global_param(cfg.timesteps)

    cfg.vis = args.planner_vis
    cfg.ik_seed_num = 8
    cfg.ik_parallel = True  # False
    mkdir_if_missing("data/demonstrations")
    scene = PlanningScene(cfg)

    for i, name in enumerate(env.obj_path[:-2]):  # load all objects
        name = name.split("/")[-1]
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()

    idx = 0
    cnt = 0
    success = 0.0

    while idx < args.num:
        print("reset scene...")
        if not env.cache_reset():
            continue

        save_scene_name = os.path.join(
            env._root_dir, args.save_dir, "scene_{}".format(idx)
        )
        observations = []

        obj_names, obj_poses = env.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        if not check_scene(object_poses):
            continue
        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):  # update planning scene
            scene.env.update_pose(name, object_poses[i])
            obj_idx = env.obj_path[:-2].index("data/objects/" + name)
            exists_ids.append(obj_idx)
            trans, orn = env.cache_object_poses[obj_idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))

        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(env.obj_path[:-2])
            if obj_idx not in exists_ids
        ]
        target_name = env.obj_path[env.target_idx].split("/")[-1]
        scene.env.set_target(target_name)         
        scene.reset(lazy=True)
        info = scene.step()
        obj_names, obj_poses = env.get_env_info()

        if len(info) > 1:
            plan = scene.planner.history_trajectories[-1]
            if cfg.vis:
                scene.fast_debug_vis()
            for k in range(plan.shape[0]):
                obs, rew, done, _ = env.step(plan[k].tolist())
                observations.append(obs)
            rew = env.retract()
            success += rew
            cnt += 1
        else:
            rew = 0

        for i, name in enumerate(object_lists[:-2]):  # reset planner
            scene.env.update_pose(name, placed_poses[i])

        print("success: {:.3f}".format(success / cnt))
        if rew > 0:  # only save successful traj
            record_traj(
                save_scene_name,
                observations,
                plan,
                obj_poses,
                obj_names,
                env._proj_matrix,
                env._view_matrix,
                target_name,
                scene.traj.goal_set,
                write_images=args.write_images
            )
            idx += 1
    env.disconnect()
