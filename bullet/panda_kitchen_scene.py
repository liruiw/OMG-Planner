# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
from gym import spaces
import time
import sys
from . import _init_paths

from omg.core import *
from omg.util import *
from omg.config import cfg
from real_world.trial import *
import pybullet as p
import numpy as np
import pybullet_data

from PIL import Image
import glob
import gym
import IPython
from panda_gripper import Panda

from transforms3d import quaternions
import scipy.io as sio
import pkgutil


class PandaKitchenEnv:
    """Class for panda cabinet environment with ycb objects.
    adapted from kukadiverse env in pybullet
    """

    def __init__(
        self,
        urdfRoot=pybullet_data.getDataPath(),
        actionRepeat=130,
        isEnableSelfCollision=True,
        renders=False,
        isDiscrete=False,
        maxSteps=800,
        dtheta=0.1,
        gui_debug=True,
        blockRandom=0.5,
        target_obj=[1, 2, 3, 4, 10, 11],  # [1,2,4,3,10,11],
        all_objs=[0, 1, 2, 3, 4, 8, 10, 11, 12, 13, 14],
        cameraRandom=0,
        width=640,
        height=480,
        numObjects=8,
        safeDistance=0.13,
        random_urdf=False,
        egl_render=False,
        cache_objects=False,
        isTest=False,
    ):
        """Initializes the pandaYCBObjectEnv.

        Args:
            urdfRoot: The diretory from which to load environment URDF's.
            actionRepeat: The number of simulation steps to apply for each action.
            isEnableSelfCollision: If true, enable self-collision.
            renders: If true, render the bullet GUI.
            isDiscrete: If true, the action space is discrete. If False, the
                action space is continuous.
            maxSteps: The maximum number of actions per episode.
            blockRandom: A float between 0 and 1 indicated block randomness. 0 is
                deterministic.
            cameraRandom: A float between 0 and 1 indicating camera placement
                randomness. 0 is deterministic.
            width: The observation image width.
            height: The observation image height.
            numObjects: The number of objects in the bin.
            isTest: If true, use the test set of objects. If false, use the train
                set of objects.
        """

        self._timeStep = 1.0 / 1000.0
        self._urdfRoot = urdfRoot
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._env_step = 0

        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._safeDistance = safeDistance
        self._root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self._p = p
        self._target_objs = target_obj
        self._all_objs = all_objs
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._shift = [0.5, 0.5, 0.5]  # to work without axis in DIRECT mode
        self._egl_render = egl_render

        self._cache_objects = cache_objects
        self._object_cached = False
        self._gui_debug = gui_debug
        self.target_idx = 0

        self.connect()

    def connect(self):
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(1.5, 300.0, -31.0, [-0.35, -0.58, -0.88])
            if not self._gui_debug:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        else:
            self.cid = p.connect(p.DIRECT)

        egl = pkgutil.get_loader("eglRenderer")
        if self._egl_render and egl:
            print('use egl renderer')
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True

    def disconnect(self):
        p.disconnect()
        self.connected = False

    def reset(self, init_joints=None, scene_file=None):
        """Environment reset"""
        # Set the camera settings.
        look = [
            -0.35,
            -0.58,
            -0.88,
        ]  # [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]] # 0.14
        distance = 1.5  # 2.5
        pitch = -31.0  # -56 + self._cameraRandom * np.random.uniform(-3, 3)
        yaw = 300  # 245 + self._cameraRandom * np.random.uniform(-3, 3)
        roll = 0
        fov = 60.0 + self._cameraRandom * np.random.uniform(-2, 2)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2
        )

        aspect = float(self._window_width) / self._window_height

        self.near = 0.1
        self.far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, self.near, self.far
        )

        # Set table and plane
        p.resetSimulation()
        p.setTimeStep(self._timeStep)

        # Intialize robot and objects
        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        self._panda = Panda(
            stepsize=self._timeStep,
            init_joints=list(init_joints),
            base_shift=self._shift,
        )
        self.obj_path, self._objectUids = self.cache_kitchen_objects(scene_file)
        if not self._object_cached:
            self._objectUids += self.cache_objects()

        self._env_step = 0
        return self._get_observation()

    def cache_objects(self):
        """
        Load all YCB objects and set up
        """
        obj_path = self._root_dir + "/data/objects/"
        ycb_objects = sorted([m for m in os.listdir(obj_path) if m.startswith("0")])
        ycb_path = ["data/objects/" + ycb_objects[i] for i in self._all_objs]

        pose = np.zeros([len(ycb_path), 3])
        pose[:, 0] = -2.0 - np.linspace(0, 4, len(ycb_path))  # place in the back
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() for p_ in ycb_path]
        objectUids = []
        self.obj_path = self.obj_path + objects_paths

        for i, name in enumerate(objects_paths):
            trans = pose[i] + np.array(pos)  # fixed position
            self.cache_object_poses_robot_coord.append(
                np.concatenate((trans, np.array(tf_quat(orn))))
            )
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(
                os.path.join(self._root_dir, name, "model_normalized.urdf"), trans, orn
            )  # xyzw
            objectUids.append(uid)
            self.cache_object_extents.append(
                np.loadtxt(
                    os.path.join(self._root_dir, name, "model_normalized.extent.txt")
                )
            )
            p.changeDynamics(
                uid,
                -1,
                restitution=0.1,
                mass=0.5,
                spinningFriction=0,
                rollingFriction=0,
                lateralFriction=0.9,
            )

        self._object_cached = True
        self.cached_objects += [False] * len(self.obj_path)
        return objectUids

    def cache_kitchen_objects(self, scene_file):

        mat = sio.loadmat(scene_file)
        object_lists = [f.strip() for f in mat["object_lists"]]  # .encode("utf-8")
        file_num = len(object_lists)
        files = range(file_num)
        file_num = len(files)
        poses = [
            mat["object_poses"][i] for i in files if not object_lists[i].startswith("0")
        ]
        object_lists = [
            "data/objects/" + o for o in object_lists if not o.startswith("0")
        ]

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() for p_ in object_lists]
        objectUids = []

        self.cache_object_poses = []
        self.cache_object_poses_robot_coord = []
        self.cache_object_extents = []
        for i, name in enumerate(objects_paths):
            pose = poses[i]
            self.cache_object_poses_robot_coord.append(pose)
            trans = pose[:3] + np.array(pos)  # fixed position
            orn = ros_quat(pose[3:])
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(
                os.path.join(self._root_dir, name, "model_normalized.urdf"),
                trans,
                orn,
                fix_base=True,
            )  # xyzw
            objectUids.append(uid)
            self.cache_object_extents.append(
                np.loadtxt(
                    os.path.join(self._root_dir, name, "model_normalized.extent.txt")
                )
            )
            for other_uid in objectUids:
                p.setCollisionFilterPair(
                    uid, other_uid, -1, -1, 0
                )  # unnecessary simulation effort

        self.cached_objects = [False] * len(objects_paths)
        return objects_paths, objectUids

    def reset_objects(self):
        for idx, obj in enumerate(self._objectUids):
            if self.cached_objects[idx]:
                p.resetBasePositionAndOrientation(
                    obj,
                    self.cache_object_poses[idx][0],
                    self.cache_object_poses[idx][1],
                )
            self.cached_objects[idx] = False

    def cache_reset(self, init_joints=None, scene_file=None):
        self._panda.reset(init_joints)
        self.reset_objects()

        if scene_file is None or not os.path.exists(scene_file):
            self._randomly_place_objects(self._get_random_object(self._numObjects))
        else:
            self.place_objects_from_scene(scene_file)
        self._env_step = 0
        self.obj_names, self.obj_poses = self.get_env_info()
        return self._get_observation()

    def place_objects_from_scene(self, scene_file):
        """place objects with pose based on the scene file"""
        scene = sio.loadmat(scene_file)

        poses = scene["object_poses"]
        path = scene["object_lists"]
        path = ["data/objects/" + o for o in path]
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objectUids = []
        objects_paths = [
            _p.strip() for _p in path if "table" not in _p and "floor" not in _p
        ]

        for i, name in enumerate(objects_paths):
            if "0" not in name:
                continue
            obj_idx = self.obj_path.index(name)
            pose = poses[i]
            trans = pose[:3] + np.array(pos) - np.array([0, 0, 0.05])  # fixed position
            orn = ros_quat(pose[3:])
            p.resetBasePositionAndOrientation(self._objectUids[obj_idx], trans, orn)
            self.cached_objects[obj_idx] = True

        self.target_idx = self.obj_path.index(objects_paths[0])
        for _ in range(2000):
            p.stepSimulation()
        return objectUids

    def close_finger(self):
        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-2:] = 0
        self.step(cur_joint.tolist())  # grasp

    def open_finger(self):
        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-2:] = 0.04
        self.step(cur_joint.tolist())  # grasp

    def step(self, action, obs=True, close_finger=False):
        """Environment step."""
        self._env_step += 1
        if close_finger:
            action[-2:] = 0.0, 0.0
        self._panda.setTargetPositions(action)
        for _ in range(self._actionRepeat):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        if not obs:
            observation = None
        else:
            observation = self._get_observation()
        done = self._termination()
        reward = self._reward()

        return observation, reward, done, None

    def _get_observation(self):
        _, _, rgba, depth, mask = p.getCameraImage(
            width=self._window_width,
            height=self._window_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            physicsClientId=self.cid,
        )

        depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        joint_pos, joint_vel = self._panda.getJointStates()
        obs = np.concatenate(
            [rgba[..., :3], depth[..., None], mask[..., None]], axis=-1
        )
        return (obs, joint_pos)

    def _get_target_obj_pose(self):
        return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]

    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        hand_pos, _ = p.getLinkState(
            self._panda.pandaUid, self._panda.pandaEndEffectorIndex
        )[:2]
        pos, _ = p.getBasePositionAndOrientation(
            self._objectUids[self.target_idx]
        )  # target object
        if (
            np.linalg.norm(np.subtract(pos, hand_pos)) < 0.2
            and pos[2] > -0.35 - self._shift[2]
        ):
            reward = 1
        return reward

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._env_step >= self._maxSteps

    def _add_mesh(self, obj_file, trans, quat, scale=1, fix_base=False):
        try:
            bid = p.loadURDF(
                obj_file, trans, quat, globalScaling=scale, useFixedBase=fix_base
            )
            return bid
        except:
            print("load {} failed".format(obj_file))

    def get_env_info(self, add_table_floor=False):
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if self.cached_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append(self.obj_path[idx])  # .encode("utf-8")

        return obj_dir, poses


def bullet_execute_plan(
    env,
    plan,
    write_video,
    video_writer,
    end_close_finger=False,
    close_finger=False,
    open_finger_step=-1,
):
    for k in range(plan.shape[0]):
        obs, rew, done, _ = env.step(plan[k].tolist(), close_finger=close_finger)
        if write_video:
            video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
        if open_finger_step >= 0 and k == plan.shape[0] - open_finger_step - 1:
            env.open_finger()
            close_finger = False
    if end_close_finger:
        env.close_finger()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="filename", type=str, default="kitchen1")
    parser.add_argument(
        "-d", "--dir", help="filename", type=str, default="data/scenes/"
    )
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    parser.add_argument("-pv", "--planner_vis", help="renders", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument(
        "-s", "--script", help="load script to run", default="script.txt"
    )
    parser.add_argument("--egl", help="use egl render", action="store_true")    
    args = parser.parse_args()
    

    # load script and file
    place_script_idx, place_poses = 0, []
    target_script_idx, target_names = 0, []
    end_script_idx, end_confs = 0, []
    target = None

    vis_once = False
    vis_collision = False
    write_video = args.write_video
    video_writer = None
    
    # setup planner
    config.cfg.report_cost = False
    config.cfg.traj_init = "grasp"
    config.cfg.scene_file = ""
    config.cfg.vis = args.planner_vis
    config.cfg.goal_set_max_num = 100
    config.cfg.get_global_param()

    if args.script is not None:
        file_name = config.cfg.root_dir + "real_world/" + args.script
        if not os.path.exists(file_name): 
            print('script not exists')
            sys.exit()
            
        text_file = open(file_name, "r" )   
        lines = text_file.readlines()
        target_names = [line[2:].strip() for line in lines if line.startswith("T")]
        place_poses = [
            [float(s) for s in line[2:].split(",")]
            for line in lines
            if line.startswith("P")
        ]
        end_confs = [
            [float(s) for s in line[2:].split(",")]
            for line in lines
            if line.startswith("E")
        ]
        if len(end_confs) == 0:
            end_confs = [" "]  # whatever
        print("loaded one task script:", args.script)
        print("target:", target_names)
        print("place delta poses:", place_poses)
        vis_once = lines[-1].strip() == "ONCE"

    # config.cfg.increment_iks = True
    config.cfg.traj_init = "grasp" 
    config.cfg.scene_file = ""
    config.cfg.dynamic_timestep = True  
    scene = PlanningScene(config.cfg)
    start_conf = np.array(
        [-0.05929, -1.6124, -0.197, -2.53, -0.0952, 1.678, 0.587, 0.0398, 0.0398]
    )

    # setup bullet env
    mkdir_if_missing('output_videos')
    config.cfg.output_video_name = "output_videos/bullet_" + args.file + ".avi"
    env = PandaKitchenEnv(renders=args.vis, egl_render=args.egl)
    scene_file = os.path.join(cfg.scene_path, args.file + ".mat")
    s = time.time()
    env.reset(start_conf, scene_file)
    print("bullet load time", time.time() - s)
    if args.write_video:
        video_writer = cv2.VideoWriter(
            config.cfg.output_video_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (640, 480),
        )
    for i, name in enumerate(env.obj_path):  # load all objects
        name = name.split("/")[-1]
        pose = env.cache_object_poses_robot_coord[i]
        scene.env.add_object(name, pose[:3], pose[3:], compute_grasp=True)

    scene.env.combine_sdfs()
    env.cache_reset(scene_file=scene_file)
    obj_names, obj_poses = env.get_env_info()
    object_lists = [name.split("/")[-1].strip() for name in obj_names]
    object_poses = [pack_pose(pose) for pose in obj_poses]
    exists_ids, placed_poses = [], []

    for i, name in enumerate(object_lists):  # update planning scene
        scene.env.update_pose(name, object_poses[i])
        obj_idx = env.obj_path.index("data/objects/" + name)
        exists_ids.append(obj_idx)
        pose = env.cache_object_poses_robot_coord[obj_idx]
        placed_poses.append(np.hstack([pose[:3], pose[3:]]))
 
    scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
    scene.reset(lazy=True)

    # pick and place
    while True:

        # grasp
        if target_script_idx < len(target_names):
            target_ = target_names[target_script_idx]
        else:
            target_ = ""
        if target_ in scene.env.names:
            target = target_
            traj = plan_to_target(scene, start_conf, target)
            bullet_execute_plan(
                env, traj, args.write_video, video_writer, end_close_finger=True
            )

        else:
            print("Set target example: 004_sugar_box")

        # placement
        if place_script_idx < len(place_poses):
            place_pose_ = place_poses[place_script_idx]
        else:
            place_pose_ = ""
        if len(place_pose_) == 4 and target in scene.env.names:
            place_pose = np.array([float(item) for item in place_pose_[:3]])
            use_standoff = float(place_pose_[-1]) == 1.0
            traj_ = place_target(
                scene,
                traj[-1],
                target,
                place_pose,
                use_standoff,
                vis_collision=vis_collision,
                vis_once=vis_once,
                write_video=write_video,
            )
            if traj_ is not None:
                traj = traj_
            if cfg.vis:
                scene.fast_debug_vis(interact=1, nonstop=True)
            open_finger_step = -1 if not use_standoff else cfg.reach_tail_length                 
            bullet_execute_plan(
                env, traj, args.write_video, video_writer, close_finger=True, open_finger_step=open_finger_step
            )
        else:
            print("Place example: Kitchen 11 0.0, -0.1, 0")

        place_script_idx += 1
        target_script_idx += 1
        end_script_idx += 1
        if (
            args.script is not None
            and target_script_idx >= len(target_names)
            and place_script_idx >= len(place_poses)
            and end_script_idx >= len(end_confs)
        ):
            break

    env.disconnect()
