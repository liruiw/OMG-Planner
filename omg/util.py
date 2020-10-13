# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import random
from scipy import interpolate
import scipy.io as sio
import IPython
import time
import cv2

np.random.seed(233)
util_anchor_seeds = np.array(
    [
        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0.04, 0.04],
        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0.04, 0.04],
        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0.04, 0.04],
        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27, 0.04, 0.04],
        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85, 0.04, 0.04],
        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27, 0.04, 0.04],
        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74, 0.04, 0.04],
        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27, 0.04, 0.04],
        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0.04, 0.04],
        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0.04, 0.04],
        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35, 0.04, 0.04],
        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24, 0.04, 0.04],
        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23, 0.04, 0.04],
    ]
)


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), 0, np.sin(rotz), 0],
            [0, 1, 0, 0],
            [-np.sin(rotz), 0, np.cos(rotz), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def deg2rad(deg):
    if type(deg) is list:
        return [x / 180.0 * np.pi for x in deg]
    return deg / 180.0 * np.pi


def rad2deg(rad):
    if type(rad) is list:
        return [x / np.pi * 180 for x in rad]
    return rad / np.pi * 180


def pose2np(pose_kdl):
    pose = np.eye(4)
    for i in range(3):
        for j in range(4):
            pose[i, j] = pose_kdl[i, j]  # save fix initial angle pose
    return pose


def quat2rotmat(quat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = quaternions.quat2mat(quat)
    return quat_mat


def mat2rotmat(mat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = mat
    return quat_mat


def quat2rotmat(quat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = quaternions.quat2mat(quat)
    return quat_mat


def safemat2quat(mat):
    quat = np.array([1, 0, 0, 0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat


def unpack_pose(pose):
    unpacked = np.eye(4)
    unpacked[:3, :3] = quat2mat(pose[3:])
    unpacked[:3, 3] = pose[:3]
    return unpacked


def pack_pose(pose):
    packed = np.zeros(7)
    packed[:3] = pose[:3, 3]
    packed[3:] = safemat2quat(pose[:3, :3])
    return packed


def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def inv_pose(pose):
    return pack_pose(np.linalg.inv(unpack_pose(pose)))


def relative_pose(pose1, pose2, unpack=False):
    if unpack:
        return pack_pose(np.linalg.inv(pose1).dot(pose2))
    else:
        return pack_pose(np.linalg.inv(unpack_pose(pose1)).dot(unpack_pose(pose2)))


def inv_relative_pose(pose1, pose2, decompose=False):

    from_pose = np.eye(4)
    from_pose[:3, :3] = quat2mat(pose1[3:])
    from_pose[:3, 3] = pose1[:3]
    to_pose = np.eye(4)
    to_pose[:3, :3] = quat2mat(pose2[3:])
    to_pose[:3, 3] = pose2[:3]
    relative_pose = np.linalg.inv(to_pose).dot(from_pose)
    return relative_pose


def compose_pose(pose1, pose2):
    return pack_pose(unpack_pose(pose1).dot(unpack_pose(pose2)))


def get_diff_matrix(
    n, diff_rules, time_interval, diff_rule_length=7, order=1, with_end=True
):
    diff_rule = diff_rules[order - 1]
    half_length = diff_rule_length // 2
    diff_matrix = np.zeros([n + 1, n])
    for i in range(0, n + 1):
        for j in range(-half_length, half_length):
            index = i + j
            if index >= 0 and index < n:
                diff_matrix[i, index] = diff_rule[j + half_length]
    if with_end == False:
        diff_matrix[-1, -1] = 0
    return diff_matrix / (time_interval ** order)


def safe_div(dividend, divisor, eps=1e-8):  # mark
    return dividend / (divisor + eps)


def wrap_value(value):
    if value.shape[0] <= 7:
        return rad2deg(value)
    value_new = np.zeros(value.shape[0] + 1)
    value_new[:7] = rad2deg(value[:7])
    value_new[8:] = rad2deg(value[7:])
    return value_new


def wrap_values(value):
    if type(value) is list:
        value = np.array(value)
    value_new = rad2deg(value)
    if value.shape[1] > 7:
        value_new = np.zeros([value.shape[0], value.shape[1] + 1])
        value_new[:, :7] = rad2deg(value[:, :7])
        value_new[:, 8:] = rad2deg(value[:, 7:])
    return value_new


def wrap_index(value):
    if value == 10:  # right finger
        return list(range(7)) + [8]
    elif value > 7:
        return list(range(value - 1))
    return list(range(value))


def wrap_joint(value):
    if value == 8:
        return list(range(7))
    if value == 9:
        return list(range(7)) + [8]
    if value == 10:
        return list(range(7)) + [9]
    return list(range(value))


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def interpolate_waypoints(waypoints, n, m, mode="cubic"):  # linear
    """
    Interpolate the waypoints using interpolation.
    """

    data = np.zeros([n, m])
    x = np.linspace(0, 1, waypoints.shape[0])
    for i in range(waypoints.shape[1]):
        y = waypoints[:, i]

        t = np.linspace(0, 1, n + 2)
        if mode == "linear":  # clamped cubic spline
            f = interpolate.interp1d(x, y, "linear")
        if mode == "cubic":  # clamped cubic spline
            f = interpolate.CubicSpline(x, y, bc_type="clamped")
        elif mode == "quintic":  # seems overkill to me
            pass
        data[:, i] = f(t[1:-1])  #
        # plt.plot(x, y, 'o', t[1:-1], data[:, i], '-') #  f(np.linspace(0, 1, 5 * n+2))
        # plt.show()
    return data


def multi_interpolate_waypoints(
    start, goals, n, m, mode="cubic", include_endpoint=True
):  # linear
    """
    Interpolate the waypoints from one start to multiple goals using interpolation.
    """
    if len(start.shape) == 1:
        start = np.tile(start, [len(goals), 1])
    waypoints = np.stack([start, goals], axis=0)
    goals = np.array(goals)  # k x m
    k = goals.shape[0]
    data = np.zeros([k, n, m])
    x = np.linspace(0, 1, waypoints.shape[0])

    for i in range(waypoints.shape[-1]):
        y = waypoints[..., i]
        t = np.linspace(0, 1, n + 2)
        if mode == "linear":  # clamped cubic spline
            f = interpolate.interp1d(x, y, "linear", axis=0)
        if mode == "cubic":  # clamped cubic spline
            f = interpolate.CubicSpline(x, y, bc_type="clamped")
        elif mode == "quintic":  # seems overkill
            pass
        data[..., i] = f(t[1:-1]).T  #

        # import matplotlib.pyplot as plt
        # plt.plot(x, y[:,0], 'o', t[1:-1], data[0, :, i], '-') #  f(np.linspace(0, 1, 5 * n+2))
        # plt.show()

    return data.reshape([-1, m])


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat


def get_hand_anchor_index_point():
    hand_anchor_points = np.array(
        [
            [0, 0, 0],
            [0.00, -0.00, 0.058],
            [0.00, -0.043, 0.058],
            [0.00, 0.043, 0.058],
            [0.00, -0.043, 0.098],
            [0.00, 0.043, 0.098],
        ]
    )
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]
    return hand_anchor_points, line_index
 
def get_sample_goals(scene, goalset, goal_idx):
    grasp_set_index = []
    grasp_set_poses = []
    display_grasp_num = 8
    selected_index = list(np.linspace(0, len(scene.traj.goal_set) - 1, display_grasp_num).astype(np.int)) + [goal_idx]
    
    for idx in selected_index:
        _, grasp_set_pose = scene.prepare_render_list(scene.traj.goal_set[idx])
        for j in range(7, 10):
            grasp_set_poses.append(grasp_set_pose[j])
        grasp_set_index.extend(list(range(7,10)))       
    return grasp_set_poses, grasp_set_index

def ycb_special_case(pose_grasp, name):
    if name == '037_scissors': # only accept top down for edge cases
        z_constraint = np.where((np.abs(pose_grasp[:, 2, 3]) > 0.09) * \
                 (np.abs(pose_grasp[:, 1, 3]) > 0.02) * (np.abs(pose_grasp[:, 0, 3]) < 0.05)) 
        pose_grasp = pose_grasp[z_constraint[0]]
        top_down = []
        
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > 0.06) 
        pose_grasp = pose_grasp[rot_constraint[0]]
    
    elif name == '024_bowl' or name == '025_mug':
        if name == '024_bowl':
            angle = 30
        else:
            angle = 15
        top_down = []
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > angle * np.pi / 180)
        pose_grasp = pose_grasp[rot_constraint[0]]
    return pose_grasp