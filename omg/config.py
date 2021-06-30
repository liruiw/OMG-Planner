# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import math
from easydict import EasyDict as edict
from .util import *
import scipy
import cv2
import sys
import torch


############ Note #############
# Parameters:
#     1.  Related to collision: target_obj_collision, epsilon, target_epsilon, clearance, top_k_collision, uncheck_finger_collision
#     2.  Related to goal selection: goal_idx, dist_eps, ol_alg, eta, normalize_cost
#     3.  Related to initialization: ik_seed_num, goal_set_max_num, remove_flip_grasp, ik_parallel, ik_clearance
#     4.  Related to optimizations: optim_steps, collision_point_num, timesteps, base_step_size, cost_schedule_boost
#     5.  Others: base_obstacle_weight, smooth_base_weight, use_standoff, dynamic_timestep, pre_terminate, standoff_dist, traj_delta


current_dir = os.path.dirname(os.path.abspath(__file__))
cfg = edict()

""" hyperparameter """
cfg.smoothness_base_weight = 0.1  # 0.1 weight for smoothness cost in total cost
cfg.base_obstacle_weight = 1.0  # 1.0 weight for obstacle cost in total cost
cfg.base_grasp_weight = 1.0  # weight for grasp cost in total cost
cfg.cost_schedule_decay = 1  # cost schedule decay for obstacle cost weight wrt base
cfg.cost_schedule_boost = 1.02  # cost schedule boost for smoothness cost weight

cfg.base_step_size = 0.1  # initial step size in gradient descent
cfg.step_decay_rate = 1.0  # decay rate for step size in gradient descent
cfg.joint_limit_max_steps = 10  # maximum smooth projection steps for joint limit
cfg.optim_steps = 50  # optimization steps for each planner call

""" planner parameters """
cfg.epsilon = 0.2  # the obstacle padding distance that has gradient
cfg.target_epsilon = 0.1  # the obstacle padding distance for target object
cfg.target_obj_collision = 0.0  # 0.0 scale factor for target object collision
cfg.collision_point_num = 15  # collision point sample number for each link
cfg.time_interval = 0.1  # time inverval for computing velocity
cfg.top_k_collision = 1000  # the number of closest point to penalize for each trajectory
cfg.link_collision_weight = np.ones([10, 1])  # link collision weight depending on size
cfg.link_smooth_weight = np.ones(9) # link smooth weight depending on size

cfg.clearance = 0.01  # clerance threshold for determining if a traj has collision
cfg.target_clearance = 0.0   # clerance threshold for determining if a traj has collision with target
cfg.ik_clearance = 0.03  # clerance threshold for determining if an ik has collision
cfg.target_size = 1.0  # target object's actual sdf resize ratio
cfg.obstacle_size = 1  # obstacle's resize ratio
cfg.obj_point_num = 800  # object points for KDTree in isf
cfg.terminate_smooth_ratio = 4  # 35 terminate condition for trajectory
cfg.terminate_grad_norm = 1.5  # terminate condition for smooth grad norm
cfg.terminate_smooth_loss = 35  # terminate condition for trajectory
cfg.penalize_constant = 5  # scaling negative sdf values to avoid collision
cfg.grasp_optimize = False  # the option of optimizing grasp term
cfg.traj_init = "grasp"  # use ik or precomputed grasp list to initialize the goal
cfg.traj_interpolate = "cubic"  # interpolate trajectory waypoints between start and end

cfg.goal_set_proj = True  # use the goal set variant of CHOMP
cfg.goal_set_max_num = 100  # the maximum number of goals in the goal set
cfg.ol_alg = "MD"  # the online learning algorithm for updating grasp distribution
cfg.dist_eps = 0.1  # weight coefficients for projected distance
cfg.goal_idx = -2  # >0: goal index in goal set, -1: closest, -2: minimum cost, -3: random
cfg.pre_terminate = True  # terminate condition
cfg.ik_seed_num = 12  # anchor seed number to solve for inverse kinematics
cfg.finger_hard_constraint = True  # direct constraint on the fingers
cfg.uncheck_finger_collision = 0  # -1 means uncheck finger collision during optimization
cfg.allow_collision_point = 5  # allowing collision point number for the entire trajectory

cfg.soft_joint_limit_padding = 0.2  # constraint planning joint limit to be smaller than actual ones
cfg.extra_smooth_steps = 20  # extra steps for postprocessing fixed goal
cfg.clip_grad_scale = 10.0  # clip the update gradients
cfg.normalize_cost = True  # normalize ol costs
cfg.disable_collision_set = []  # object names to fully disable collision check
cfg.use_standoff = True  # use use_standoff for grasp stability
cfg.standoff_dist = 0.08  # standoff distance before grasping
cfg.remove_flip_grasp = True  # remove all grasps that require wrist to rotate over 180 degrees
cfg.remove_base_rotate_grasp = True  # remove all grasps that require base to rotate over 40 degrees
cfg.remove_camera_downward_grasp = True  # remove all grasps that require base to rotate over 40 degrees

cfg.augment_flip_grasp = True  # augment grasps by flipping over 180 degrees
cfg.target_hand_filter_angle = 120  # for filtering grasps that require heavy rotation
cfg.dynamic_timestep = False  # for dynamically choosing length of the trajectory
cfg.post_standoff = False  # standoff interpolation after planning
cfg.consider_finger = False  # consider finger in optimization
cfg.reach_tail_length = 5  # the trajectory length for standoff reaching
cfg.use_layer = True  # use sdf layer or torch cuda for collision check
cfg.increment_iks = False  # for more goals during solving iks
cfg.ik_parallel = True  # faster solving iks
cfg.traj_delta = 0.05  # resolution for trajectory state discretization
cfg.colored_gripper = False  # use colored robot model
cfg.traj_max_step = 50  # maximum step
cfg.traj_min_step = 2  # minimum step
cfg.default_lazy = True # lazy grasp computation
cfg.y_upsample = False # upsampling grasps by tilting around the antipodal contacts
cfg.z_upsample = True # upsampling grasps in gravity direction for placing
cfg.use_point_sdf = False # use SDF from point cloud perception instead of object model and pose
cfg.robot_vis_offset = True # visualization offset for mesh center

""" global parameter """
cfg.root_dir = current_dir + "/../"  # root directory
cfg.exp_dir = cfg.root_dir + "output"  # export output directory
cfg.shapenet_path = cfg.root_dir + "data/ShapeNet"  # shapenet root in data directory
cfg.robot_model_path = cfg.root_dir + "data/robots"  # robot model directory
cfg.grasp_path = cfg.root_dir + "data/grasps/"  # scene file directory
cfg.scene_path = cfg.root_dir + "data/scenes/"  # scene file directory
cfg.window_height = 480  # 960 # window width for visualization
cfg.window_width = 640  # 1280 # window height for visualization
cfg.timesteps = 30  # discretize uniform timesteps for trajectory
cfg.base_link = "panda_link0"  # the base of the robot
cfg.report_cost = False  # print out cost in terminal
cfg.vis = True  # visualization option
cfg.view_init = False  # visualize initial configuration to compare
cfg.scene_file = "scene_35"  # the scene file to load
cfg.cam_pos = [
    0.63850115,
    0.5352779,
    1.4893766,
]  # the static camera position for visualization
cfg.cam_V = None  # the extrinsics for visualization
cfg.report_time = False  # verbose plan time check
cfg.output_video_name = "test_video.avi" # output video
cfg.silent = False # mute
cfg.timeout = 3. # -1.
cfg.external_grasps = None # grasps from external sources such as offline scenes or grasp detectors

""" global function """
def get_derivative(data, start, end, diff_rule=1):
    """
    Assume take difference of the last two dimensions,
    take difference to approximate derivative based on the
    config difference rule coefficients.
    """
    diff_mat = cfg.diff_matrices[diff_rule - 1]
    diff_mat = diff_mat[: data.shape[-2] + 1, : data.shape[-2]]
    data_dot = np.matmul(diff_mat, data)
    idx_middle = int(cfg.diff_rule_length / 2)
    data_dot[..., 0, :] += (
        cfg.diff_rule[diff_rule - 1][idx_middle - 1]
        * start
        / (cfg.time_interval ** diff_rule)
    )
    data_dot[..., -2, :] += (
        cfg.diff_rule[diff_rule - 1][idx_middle + 1]
        * end
        / (cfg.time_interval ** diff_rule)
    )
    data_dot[..., -1, :] += (
        cfg.diff_rule[diff_rule - 1][idx_middle]
        * end
        / (cfg.time_interval ** diff_rule)
    )
    return data_dot[..., :-1, :]  #


def get_derivative_torch(data, start, end, diff_rule=1):
    """
    Assume take difference of the last two dimensions,
    take difference to approximate derivative based on the
    config difference rule coefficients.
    """
    diff_mat = cfg.diff_matrices_torch[diff_rule - 1]
    diff_mat = diff_mat[: data.shape[-2] + 1, : data.shape[-2]]
    data_dot = torch.matmul(diff_mat, data)
    idx_middle = int(cfg.diff_rule_length / 2)
    data_dot[..., 0, :] += (
        cfg.diff_rule_torch[diff_rule - 1][idx_middle - 1]
        * start
        / (cfg.time_interval ** diff_rule)
    )
    data_dot[..., -2, :] += (
        cfg.diff_rule_torch[diff_rule - 1][idx_middle + 1]
        * end
        / (cfg.time_interval ** diff_rule)
    )
    data_dot[..., -1, :] += (
        cfg.diff_rule_torch[diff_rule - 1][idx_middle]
        * end
        / (cfg.time_interval ** diff_rule)
    )
    return data_dot[..., :-1, :]  #


def make_video_writer(name, ratio=-1):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG
    if ratio == -1:
        ratio = 2 if cfg.view_init else 1
    return cv2.VideoWriter(
        name, fourcc, 10.0, (int(ratio) * cfg.window_width, cfg.window_height)
    )


def get_global_param(steps=cfg.timesteps):
    """ global variable """
    cfg.time_interval = (0.1 * cfg.timesteps) / steps
    cfg.timesteps = steps

    cfg.diff_rule_length = 7
    cfg.diff_rule = np.array(
        [[0, 0, -1, 1, 0, 0, 0], [0, 0, 1, -2, 1, 0, 0], [0, -0.5, 1, 0, -1, 0.5, 0]]
    )
    cfg.diff_matrices = [
        get_diff_matrix(
            steps,
            cfg.diff_rule,
            cfg.time_interval,
            cfg.diff_rule_length,
            i + 1,
            not cfg.goal_set_proj,
        )
        for i in range(cfg.diff_rule.shape[0])
    ]
    cfg.A = cfg.diff_matrices[0].T.dot(cfg.diff_matrices[0])
    cfg.Ainv = np.linalg.inv(cfg.A)

    cfg.diff_rule_torch = torch.from_numpy(cfg.diff_rule).cuda().float()
    cfg.diff_matrices_torch = (
        torch.from_numpy(np.array(cfg.diff_matrices)).cuda().float()
    )
    cfg.A_torch = torch.from_numpy(cfg.A).cuda().float()
    cfg.Ainv_torch = torch.from_numpy(cfg.Ainv).cuda().float()


def get_global_path():
    cfg.shapenet_path = (
        cfg.root_dir + "data/ShapeNet"
    )  # shapenet root in data directory
    cfg.robot_model_path = cfg.root_dir + "data/robots"  # robot model directory
    cfg.grasp_path = cfg.root_dir + "data/grasps/"  # scene file directory
    cfg.scene_path = cfg.root_dir + "data/scenes/"  # scene file directory


def cfg_update(attr, val):
    cfg.attr = val


cfg.get_derivative = get_derivative
cfg.get_derivative_torch = get_derivative_torch
cfg.make_video_writer = make_video_writer
cfg.get_global_param = get_global_param
cfg.get_global_path = get_global_path
get_global_param()

