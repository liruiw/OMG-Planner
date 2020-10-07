# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import scipy.io as sio
import os

from numpy.linalg import inv
import torch
import cv2
import argparse

import IPython

import platform

PYTHON2 = True
if platform.python_version().startswith("3"):
    PYTHON2 = False
# if PYTHON2:

from . import _init_paths
import PyKDL
from .kdl_parser import kdl_tree_from_urdf_model
from .urdf_parser_py.urdf import URDF

np.random.seed(233)


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def rotz(joints):
    M = np.tile(np.eye(4), [joints.shape[0], 1, 1])
    M[..., 0, 0] = np.cos(joints)
    M[..., 0, 1] = -np.sin(joints)
    M[..., 1, 0] = np.sin(joints)
    M[..., 1, 1] = np.cos(joints)
    return M


def DH(pose, joints, offset=0):
    rotx = rotX(offset)
    M = rotz(joints)
    pose = np.matmul(pose, np.matmul(M, rotx))
    return pose


def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = PyKDL.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl


def deg2rad(deg):
    if type(deg) is list:
        return [x / 180.0 * np.pi for x in deg]
    return deg / 180.0 * np.pi


def rad2deg(rad):
    if type(rad) is list:
        return [x / np.pi * 180 for x in rad]
    return rad / np.pi * 180


class robot_kinematics(object):
    """
    Robot Kinematics built with PyKDL.
    It's mainly used to train a vision system for 7-dof robot arm with end effectors.
    """

    def __init__(self, robot, base_link=None, tip_link=None, data_path="../.."):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if PYTHON2:
            import cPickle as pickle

            with open(cur_path + "/robot.pkl", "rb") as fid:
                robot_info = pickle.load(fid)
        else:
            import pickle

            with open(cur_path + "/robot_p3.pkl", "rb") as fid:
                robot_info = pickle.load(fid)

        self._pose_0 = robot_info["_pose_0"]
        self.finger_pose = self._pose_0[-2].copy()

        self._joint_origin = robot_info["_joint_axis"]
        self._tip2joint = robot_info["_tip2joint"]
        self._joint_axis = robot_info["_joint_axis"]
        self._joint_limits = robot_info["_joint_limits"]
        self._joint2tips = robot_info["_joint2tips"]
        self._joint_name = robot_info["_joint_name"]
        self.center_offset = np.array(robot_info["center_offset"])
        self._link_names = robot_info["_link_names"]

        self._base_link, self._tip_link = "panda_link0", "panda_hand"
        self._num_jnts = 7
        self._robot = URDF.from_xml_string(
            open(
                os.path.join(cur_path, data_path, "data/robots", "panda_arm_hand.urdf"),
                "r",
            ).read()
        )

        self._kdl_tree, _ = kdl_tree_from_urdf_model(self._robot)
        self.soft_joint_limit_padding = 0.2

        mins_kdl = joint_list_to_kdl(
            np.array(
                [
                    self._joint_limits[n][0] + self.soft_joint_limit_padding
                    for n in self._joint_name[:-3]
                ]
            )
        )
        maxs_kdl = joint_list_to_kdl(
            np.array(
                [
                    self._joint_limits[n][1] - self.soft_joint_limit_padding
                    for n in self._joint_name[:-3]
                ]
            )
        )

        self._arm_chain = self._kdl_tree.getChain(self._base_link, self._tip_link)
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR_JL(
            self._arm_chain, mins_kdl, maxs_kdl, self._fk_p_kdl, self._ik_v_kdl
        )

    def forward_kinematics_parallel(
        self,
        joint_values=None,
        base_link="right_arm_mount",
        base_pose=None,
        offset=True,
        return_joint_info=False,
    ):
        """
        Input joint angles in degrees, output poses list in robot coordinates
        For a batch of joint requests
        """

        n, q = joint_values.shape
        initial_pose = np.array(self._pose_0)
        joints = deg2rad(joint_values)
        pose = np.tile(initial_pose, [n, 1, 1, 1])
        output_pose = np.zeros_like(pose)
        offsets = [0, -np.pi, np.pi, np.pi, -np.pi, np.pi, np.pi]

        cur_pose = base_pose
        if cur_pose is None:
            cur_pose = np.eye(4)
        cur_pose = cur_pose[None, ...]

        for i in range(7):
            b = DH(pose[:, i], joints[:, i], offsets[i])
            if i > 0:
                b[..., [1, 2]] *= -1

            cur_pose = np.matmul(cur_pose, b)
            output_pose[:, i] = cur_pose.copy()

        left_finger_pose = np.tile(initial_pose[8], [n, 1, 1])
        left_finger_pose[:, 1, 3] += joints[:, -2]
        right_finger_pose = np.tile(initial_pose[9], [n, 1, 1])
        right_finger_pose[:, 1, 3] -= joints[:, -1]

        output_pose[:, 7] = np.matmul(output_pose[:, 6], initial_pose[7])
        output_pose[:, 8] = np.matmul(output_pose[:, 7], left_finger_pose)
        output_pose[:, 9] = np.matmul(output_pose[:, 7], right_finger_pose)

        if return_joint_info:
            pose2axis = np.array(self._joint_axis)
            pose2origin = np.array(self._joint_origin)
            joint_pose = np.matmul(
                output_pose, self._tip2joint
            )  # pose_joint.dot(poses[idx])
            joint_axis = np.matmul(joint_pose[..., :3, :3], pose2axis[..., None])[
                ..., 0
            ]
            joint_origin = (
                np.matmul(joint_pose[..., :3, :3], pose2origin[..., None])[..., 0]
                + joint_pose[..., :3, 3]
            )

        if offset:  # for on
            output_pose = np.matmul(output_pose, self.center_offset)

        if return_joint_info:
            return output_pose, joint_origin, joint_axis
        else:
            return output_pose

    def inverse_kinematics(self, position, orientation=None, seed=None):
        """
        Inverse kinematics in radians
        """

        pos = PyKDL.Vector(position[0], position[1], position[2])
        if orientation is not None:
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(
                orientation[0], orientation[1], orientation[2], orientation[3]
            )
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed is None:
            seed = np.zeros(7)
            seed = seed[: self._num_jnts]
            seed = deg2rad(seed)
        seed_array.resize(len(seed))
        for idx in range(seed.shape[0]):
            seed_array[idx] = seed[idx]

        # Make IK Call
        if orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = np.array(list(result_angles))
            return result
        else:
            return None


if __name__ == "__main__":
    robot_kinematics("panda_arm")
