# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .optimizer import Optimizer
from .cost import Cost
from .util import *
from .online_learner import Learner
import IPython

from . import config
import time
import torch
import multiprocessing


def solve_one_pose_ik(input):
    """
    solve for one ik
    """
    (
        end_pose,
        standoff_grasp,
        one_trial,
        init_seed,
        attached,
        reach_tail_len,
        ik_seed_num,
        use_standoff,
        seeds,
    ) = input

    r = config.cfg.ROBOT
    joint = 0.04
    finger_joint = np.array([joint, joint])
    finger_joints = np.tile(finger_joint, (reach_tail_len, 1))
    reach_goal_set = []
    standoff_goal_set = []
    any_ik = False

    for seed in seeds:
        if use_standoff:
            standoff_pose = pack_pose(standoff_grasp[-1])
            standoff_ik = r.inverse_kinematics(
                standoff_pose[:3], ros_quat(standoff_pose[3:]), seed=seed
            )  #
            standoff_iks = [standoff_ik]  # this one can often be off

            if standoff_ik is not None:
                for k in range(reach_tail_len):
                    standoff_pose = pack_pose(standoff_grasp[k])
                    standoff_ik_k = r.inverse_kinematics(
                        standoff_pose[:3],
                        ros_quat(standoff_pose[3:]),
                        seed=standoff_iks[-1],
                    )  #
                    if standoff_ik_k is not None:
                        standoff_iks.append(np.array(standoff_ik_k))
                    else:
                        break
            standoff_iks = standoff_iks[1:]

            if len(standoff_iks) == reach_tail_len:
                if not attached:
                    standoff_iks = standoff_iks[::-1]
                reach_traj = np.stack(standoff_iks)
                diff = np.linalg.norm(np.diff(reach_traj, axis=0))

                if diff < 2:  # smooth
                    standoff_ = standoff_iks[0] if not attached else standoff_iks[-1]
                    reach_traj = np.concatenate([reach_traj, finger_joints], axis=-1)
                    reach_goal_set.append(reach_traj)
                    standoff_goal_set.append(
                        np.concatenate([standoff_, finger_joint])
                    )  # [-1]
                    any_ik = True

        else:
            goal_ik = r.inverse_kinematics(
                end_pose[:3], ros_quat(end_pose[3:]), seed=seed
            )
            if goal_ik is not None:
                reach_goal_set.append(np.concatenate([goal_ik, finger_joint]))
                standoff_goal_set.append(np.concatenate([goal_ik, finger_joint]))
                any_ik = True
    return reach_goal_set, standoff_goal_set, any_ik


class Planner(object):
    """
    Planner class that plans a grasp trajectory
    Tricks such as standoff pregrasp, flip grasps are for real world experiments.
    """

    def __init__(self, env, traj, lazy=False):

        self.cfg = config.cfg  # env.config
        self.env = env
        self.traj = traj
        self.cost = Cost(env)
        self.optim = Optimizer(env, self.cost)
        self.lazy = lazy

        if self.cfg.goal_set_proj:
            if hasattr(self.cfg, 'use_external_grasp') and self.cfg.use_external_grasp:
                self.load_goal_from_external(self.cfg.external_grasps)

            if self.cfg.scene_file == "" or self.cfg.traj_init == "grasp":
                self.load_grasp_set(env)
                self.setup_goal_set(env)
            else:
                self.load_goal_from_scene()

            self.grasp_init(env)
            self.learner = Learner(env, traj, self.cost)
        else:
            self.traj.interpolate_waypoints()
        self.history_trajectories = []
        self.info = []
        self.ik_cache = []

    # update planner based on the env
    def update(self, env, traj):
        self.cfg = config.cfg
        self.env = env
        self.traj = traj

        # update cost
        self.cost.env = env
        self.cost.cfg = config.cfg
        if len(self.env.objects) > 0:
            self.cost.target_obj = self.env.objects[self.env.target_idx]

        # update optimizer
        self.optim = Optimizer(env, self.cost)

        # load grasps if needed
        if self.cfg.goal_set_proj:
            if hasattr(self.cfg, 'use_external_grasp') and self.cfg.use_external_grasp:
                self.load_goal_from_external(self.cfg.external_grasps)
            elif self.cfg.scene_file == "" or self.cfg.traj_init == "grasp":
                self.load_grasp_set(env)
                self.setup_goal_set(env)
            else:
                self.load_goal_from_scene()

            self.grasp_init(env)
            self.learner = Learner(env, traj, self.cost)
        else:
            self.traj.interpolate_waypoints()
        self.history_trajectories = []
        self.info = []
        self.ik_cache = []

    def load_goal_from_scene(self):
        """
        Load saved goals from scene file, standoff is not used.
        """
        file = os.path.join(self.cfg.scene_path, self.cfg.scene_file + ".mat")
        if self.cfg.traj_init == "scene" and not hasattr(self.cfg, 'force_standoff'):
            self.cfg.use_standoff = False

        if os.path.exists(file):
            scene = sio.loadmat(file)
            self.cfg.goal_set_max_num = len(scene["goals"])
            indexes = range(self.cfg.goal_set_max_num)
            self.traj.goal_set = scene["goals"][indexes]
            self.env.objects[self.env.target_idx].reach_grasps = scene["reach_grasps"][indexes]
            if "grasp_qualities" in scene:
                self.traj.goal_quality = scene["grasp_qualities"][0][indexes]
                self.traj.goal_potentials = scene["grasp_potentials"][0][indexes]
            else:
                self.traj.goal_quality = np.zeros(self.cfg.goal_set_max_num)
                self.traj.goal_potentials = np.zeros(self.cfg.goal_set_max_num)

    def load_goal_from_external(self, grasp_list):
        """
        Load grasps detected by other methods.
        """

        """ external grasps poses """
        target_obj = self.env.objects[self.env.target_idx]
        pose_grasp = np.array(grasp_list)
        self.solve_and_process_ik(target_obj, pose_grasp, False, obj_coord=False)
        target_obj.compute_grasp = True
        self.setup_goal_set(self.env ) #

    def grasp_init(self, env=None):
        """
        Use precomputed grasps to initialize the end point and goal set
        """

        if self.cfg.scene_file == "" or self.cfg.traj_init == "grasp":
            if len(env.objects) > 0:
                self.traj.goal_set = env.objects[env.target_idx].grasps
                self.traj.goal_potentials = env.objects[env.target_idx].grasp_potentials
                if self.cfg.goal_set_proj and self.cfg.use_standoff:
                    if len(env.objects[env.target_idx].reach_grasps) > 0:
                        self.traj.goal_set = env.objects[env.target_idx].reach_grasps[:, -1]

        if len(self.traj.goal_set) > 0:
            proj_dist = np.linalg.norm(
                (self.traj.start - np.array(self.traj.goal_set))
                * self.cfg.link_smooth_weight,
                axis=-1,
            )
            self.traj.goal_quality = np.ones(len(self.traj.goal_set))

            if self.cfg.goal_idx >= 0: # manual specify
                self.traj.goal_idx = self.cfg.goal_idx

            elif self.cfg.goal_idx == -1:  # initial
                costs = self.traj.goal_potentials + self.cfg.dist_eps * proj_dist
                self.traj.goal_idx = np.argmin(costs)

            else:
                self.traj.goal_idx = 0

            if self.cfg.ol_alg == "Proj":  #
                self.traj.goal_idx = np.argmin(proj_dist)

            self.traj.end = self.traj.goal_set[self.traj.goal_idx]  #
            self.traj.interpolate_waypoints()


    def flip_grasp(self, old_grasps):
        """
        flip wrist in joint space for augmenting symmetry grasps
        """
        grasps = np.array(old_grasps[:])
        neg_mask, pos_mask = (grasps[..., -3] < 0), (grasps[..., -3] > 0)
        grasps[neg_mask, -3] += np.pi
        grasps[pos_mask, -3] -= np.pi
        limits = (grasps[..., -3] < 2.8973 - self.cfg.soft_joint_limit_padding) * (
            grasps[..., -3] > -2.8973 + self.cfg.soft_joint_limit_padding
        )
        return grasps, limits

    def solve_and_process_ik(self, target_obj, pose_grasp, z_upsample, obj_coord=True):
        env = self.env
        target_obj.reach_grasps, target_obj.grasps = self.solve_goal_set_ik(
            target_obj, env, pose_grasp, z_upsample=z_upsample, y_upsample=self.cfg.y_upsample, obj_coord=obj_coord
        )
        target_obj.grasp_potentials = []

        if (
            self.cfg.augment_flip_grasp
            and not target_obj.attached
            and len(target_obj.reach_grasps) > 0
        ):
            """ add augmenting symmetric grasps in C space """
            flip_grasps, flip_mask = self.flip_grasp(target_obj.grasps)
            flip_reach, flip_reach_mask = self.flip_grasp( target_obj.reach_grasps )
            mask = flip_mask
            target_obj.reach_grasps.extend(list(flip_reach[mask]))
            target_obj.grasps.extend(list(flip_grasps[mask]))
        target_obj.reach_grasps = np.array(target_obj.reach_grasps)
        target_obj.grasps = np.array(target_obj.grasps)

        if (
            self.cfg.remove_flip_grasp
            and len(target_obj.reach_grasps) > 0
            and not target_obj.attached
        ):
            """ remove grasps in task space that have large rotation change """
            start_hand_pose =  self.env.robot.robot_kinematics.forward_kinematics_parallel(
                                wrap_value(self.traj.start)[None] )[0][7]

            if self.cfg.use_standoff:
                n = 5
                interpolated_traj = multi_interpolate_waypoints(
                    self.traj.start, np.array(target_obj.reach_grasps[:, -1]), n, 9, "linear" )
                target_hand_pose =  self.env.robot.robot_kinematics.forward_kinematics_parallel(
                                    wrap_values(interpolated_traj))[:, 7]

                target_hand_pose = target_hand_pose.reshape(-1, n, 4, 4)
            else:
                target_hand_pose =  self.env.robot.robot_kinematics.forward_kinematics_parallel(
                                    wrap_values(np.array(target_obj.grasps)))[:, 7]

            if len(target_hand_pose.shape) == 3:
                target_hand_pose = target_hand_pose[:,None]

            # difference angle
            R_diff = np.matmul(target_hand_pose[..., :3, :3], start_hand_pose[:3,:3].transpose(1,0))
            angle = np.abs(np.arccos((np.trace(R_diff, axis1=2, axis2=3) - 1 ) /  2))
            angle = angle * 180 / np.pi
            rot_masks = angle > self.cfg.target_hand_filter_angle
            z = target_hand_pose[..., :3, 0] / np.linalg.norm(target_hand_pose[..., :3, 0], axis=-1, keepdims=True)
            downward_masks = z[:,:,-1] < -0.3
            masks = (rot_masks + downward_masks).sum(-1) > 0
            target_obj.reach_grasps = list(target_obj.reach_grasps[~masks])
            target_obj.grasps = list(target_obj.grasps[~masks])


    def solve_goal_set_ik(
        self, target_obj, env, pose_grasp, one_trial=False, z_upsample=False, y_upsample=False, obj_coord=True
    ):
        """
        Solve the IKs to the goals
        """

        object_pose = unpack_pose(target_obj.pose)
        start_time = time.time()
        init_seed = self.traj.start[:7]
        reach_tail_len = self.cfg.reach_tail_length
        reach_goal_set = []
        standoff_goal_set = []
        reach_traj_set = []
        cnt = 0
        anchor_seeds = util_anchor_seeds[: self.cfg.ik_seed_num].copy()

        if one_trial == True:
            seeds = init_seed[None, :]
        else:
            seeds = np.concatenate([init_seed[None, :], anchor_seeds[:, :7]], axis=0)

        """ IK prep """
        if obj_coord:
            pose_grasp_global = np.matmul(object_pose, pose_grasp)  # gripper -> object
        else:
            pose_grasp_global = pose_grasp

        if z_upsample:
            # Added upright/gravity (support from base for placement) upsampling by object global z rotation
            bin_num = 50
            global_rot_z = np.linspace(-np.pi, np.pi, bin_num)
            global_rot_z = np.stack([rotZ(z_ang) for z_ang in global_rot_z], axis=0)
            translation = object_pose[:3, 3]
            pose_grasp_global[:, :3, 3] = (
                pose_grasp_global[:, :3, 3] - object_pose[:3, 3]
            )  # translate to object origin
            pose_grasp_global = np.matmul(global_rot_z, pose_grasp_global)  # rotate
            pose_grasp_global[:, :3, 3] += translation  # translate back

        if y_upsample:
            # Added upsampling by local y rotation around finger antipodal contact
            bin_num = 10
            global_rot_y = np.linspace(-np.pi / 4, np.pi / 4, bin_num)
            global_rot_y = np.stack([rotY(y_ang) for y_ang in global_rot_y], axis=0)
            finger_translation = pose_grasp_global[:, :3, :3].dot(np.array([0, 0, 0.13])) + pose_grasp_global[:, :3, 3]
            local_rotation = np.matmul(pose_grasp_global[:, :3, :3], global_rot_y[:, None, :3, :3])
            delta_translation  = local_rotation.dot(np.array([0, 0, 0.13]))
            pose_grasp_global = np.tile(pose_grasp_global[:,None], (1, bin_num, 1, 1))
            pose_grasp_global[:,:,:3,3]  = (finger_translation[None] - delta_translation).transpose((1,0,2))
            pose_grasp_global[:,:,:3,:3] = local_rotation.transpose((1,0,2,3))
            pose_grasp_global = pose_grasp_global.reshape(-1, 4, 4)

        # standoff
        pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1, 1))
        if self.cfg.use_standoff:
            pose_standoff[:, 0, 2, 3] = -self.cfg.standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)

        standoff_grasp_global = np.matmul(pose_grasp_global, pose_standoff)
        parallel = self.cfg.ik_parallel
        seeds_ = seeds[:]

        if not parallel: # solve IK in sequence
            hand_center = np.empty((0, 3))
            for grasp_idx in range(pose_grasp_global.shape[0]):
                end_pose = pack_pose(pose_grasp_global[grasp_idx])
                if (
                    len(standoff_goal_set) > 0
                    and len(hand_center) > 0
                    and self.cfg.increment_iks
                ):  # augment
                    dists = np.linalg.norm(end_pose[:3] - hand_center, axis=-1)
                    closest_idx, _ = np.argsort(dists)[:1], np.amin(dists)
                    seeds_ = np.concatenate(
                            [seeds, np.array(standoff_goal_set)[closest_idx, :7].reshape(-1, 7) ], axis=0 )

                standoff_pose = standoff_grasp_global[:, grasp_idx]
                reach_goal_set_i, standoff_goal_set_i, any_ik = solve_one_pose_ik(
                    [
                        end_pose,
                        standoff_pose,
                        one_trial,
                        init_seed,
                        target_obj.attached,
                        self.cfg.reach_tail_length,
                        self.cfg.ik_seed_num,
                        self.cfg.use_standoff,
                        seeds_,
                    ]
                )
                reach_goal_set.extend(reach_goal_set_i)
                standoff_goal_set.extend(standoff_goal_set_i)

                if not any_ik:
                    cnt += 1
                else:
                    hand_center = np.concatenate([hand_center,
                                np.tile(end_pose[:3], (len(standoff_goal_set_i), 1)) ], axis=0 )

        else:
            processes = 4
            reach_goal_set = (
                np.zeros([0, self.cfg.reach_tail_length, 9])
                if self.cfg.use_standoff
                else np.zeros([0, 9])
            )
            standoff_goal_set = np.zeros([0, 9])
            any_ik, cnt = [], 0
            p = multiprocessing.Pool(processes=processes)

            num = pose_grasp_global.shape[0]
            for i in range(0, num, processes):
                param_list = [
                    [
                        pack_pose(pose_grasp_global[idx]),
                        standoff_grasp_global[:, idx],
                        one_trial,
                        init_seed,
                        target_obj.attached,
                        self.cfg.reach_tail_length,
                        self.cfg.ik_seed_num,
                        self.cfg.use_standoff,
                        seeds_,
                    ]
                    for idx in range(i, min(i + processes, num - 1))
                ]

                res = p.map(solve_one_pose_ik, param_list)
                any_ik += [s[2] for s in res]

                if np.sum([s[2] for s in res]) > 0:
                    reach_goal_set = np.concatenate(
                        ( reach_goal_set,
                           np.concatenate( [np.array(s[0]) for s in res if len(s[0]) > 0], axis=0)), axis=0)
                    standoff_goal_set = np.concatenate(
                        ( standoff_goal_set,
                          np.concatenate( [s[1] for s in res if len(s[1]) > 0], axis=0 ), ),
                        axis=0,
                    )

                if self.cfg.increment_iks:
                    max_index = np.random.choice(
                        np.arange(len(standoff_goal_set)),
                        min(len(standoff_goal_set), 10),
                    )
                    seeds_ = np.concatenate( (seeds, standoff_goal_set[max_index, :7]) )
            p.terminate()
            cnt = np.sum(1 - np.array(any_ik))

        if not self.cfg.silent:
            print(
            "{} IK init time: {:.3f}, failed_ik: {}, goal set num: {}/{}".format(
                target_obj.name,
                time.time() - start_time,
                cnt,
                len(reach_goal_set),
                pose_grasp_global.shape[0],
            )
        )
        return list(reach_goal_set), list(standoff_goal_set)

    def load_grasp_set(self, env):
        """
        Example to load precomputed grasps for YCB Objects.
        """
        for i, target_obj in enumerate(env.objects):

            if target_obj.compute_grasp and (i == env.target_idx or not self.lazy):

                if not target_obj.attached:

                    """ simulator generated poses """
                    if len(target_obj.grasps_poses) == 0:
                        simulator_path = (
                            self.cfg.robot_model_path
                            + "/../grasps/simulated/{}.npy".format(target_obj.name)
                        )
                        if not os.path.exists(simulator_path):
                            continue
                        try:
                            simulator_grasp = np.load(simulator_path, allow_pickle=True)
                            pose_grasp = simulator_grasp.item()["transforms"]
                        except:
                            simulator_grasp = np.load(
                                simulator_path,
                                allow_pickle=True,
                                fix_imports=True,
                                encoding="bytes",
                            )
                            pose_grasp = simulator_grasp.item()[b"transforms"]

                        offset_pose = np.array(rotZ(np.pi / 2))  # and
                        pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
                        pose_grasp = ycb_special_case(pose_grasp, target_obj.name)
                        target_obj.grasps_poses = pose_grasp

                    else:
                        pose_grasp = target_obj.grasps_poses
                    z_upsample = False

                else:  # placement
                    pose_grasp = np.linalg.inv(unpack_pose(target_obj.rel_hand_pose))[ None ]
                    z_upsample = self.cfg.z_upsample

                self.solve_and_process_ik(target_obj, pose_grasp, z_upsample)

    def setup_goal_set(self, env, filter_collision=True, filter_diversity=True):
        """
        Remove the goals that are in collision
        """
        """ collision """

        for i, target_obj in enumerate(env.objects):
            goal_set = target_obj.grasps
            reach_goal_set = target_obj.reach_grasps
            if len(goal_set) > 0 and target_obj.compute_grasp:  # goal_set
                potentials, _, vis_points, collide = self.cost.batch_obstacle_cost(
                    goal_set, special_check_id=i, uncheck_finger_collision=-1
                )  # n x (m + 1) x p (x 3)

                threshold = (
                    0.5
                    * (self.cfg.epsilon - self.cfg.ik_clearance) ** 2
                    / self.cfg.epsilon
                )  #
                collide = collide.sum(-1).sum(-1).detach().cpu().numpy()
                potentials = potentials.sum(dim=(-2, -1)).detach().cpu().numpy()
                ik_goal_num = len(goal_set)

                if filter_collision:
                    collision_free = (
                        collide <= self.cfg.allow_collision_point
                    ).nonzero()  # == 0

                    new_goal_set = []
                    ik_goal_num = len(goal_set)
                    goal_set = [goal_set[idx] for idx in collision_free[0]]
                    try:
                        reach_goal_set = [reach_goal_set[idx] for idx in collision_free[0]]
                    except:
                        pass
                    potentials = potentials[collision_free[0]]
                    vis_points = vis_points[collision_free[0]]


                """ diversity """
                diverse = False
                sample = False
                num = len(goal_set)
                indexes = range(num)

                if filter_diversity:
                    if num > 0:
                        diverse = True
                        unique_grasps = [goal_set[0]]  # diversity
                        indexes = []

                        for j, joint in enumerate(goal_set):
                            dists = np.linalg.norm(
                                np.array(unique_grasps) - joint, axis=-1
                            )
                            min_dist = np.amin(dists)
                            if min_dist < 0.5:  # 0.01
                                continue
                            unique_grasps.append(joint)
                            indexes.append(j)
                        num = len(indexes)

                    """ sample """
                if num > 0:
                    sample = True
                    sample_goals = np.random.choice(
                        indexes, min(num, self.cfg.goal_set_max_num), replace=False )

                    target_obj.grasps = [goal_set[int(idx)] for idx in sample_goals]
                    target_obj.reach_grasps = [
                        reach_goal_set[int(idx)] for idx in sample_goals
                    ]
                    target_obj.seeds += target_obj.grasps
                    # compute 5 step interpolation for final reach
                    target_obj.reach_grasps = np.array(target_obj.reach_grasps)
                    target_obj.grasp_potentials.append(potentials[sample_goals])
                    target_obj.grasp_vis_points.append(vis_points[sample_goals])
                    if not self.cfg.silent:
                        print(
                        "{} IK FOUND collision-free goal num {}/{}/{}/{}".format(
                            env.objects[i].name,
                            len(target_obj.reach_grasps),
                            len(target_obj.grasps),
                            num,
                            ik_goal_num,
                            )
                        )
                else:
                    print("{} IK FAIL".format(env.objects[i].name))

                if not sample:
                    target_obj.grasps = []
                    target_obj.reach_grasps = []
                    target_obj.grasp_potentials = []
                    target_obj.grasp_vis_points = []
            target_obj.compute_grasp = False


    def plan(self, traj):
        """
        Run optimizer to do trajectory optmization
        """

        self.history_trajectories = [np.copy(traj.data)]
        self.info = []
        self.selected_goals = []
        start_time_ = time.time()
        alg_switch = self.cfg.ol_alg != "Baseline" and self.cfg.ol_alg != "Proj"

        if (not self.cfg.goal_set_proj) or len(self.traj.goal_set) > 0:
            for t in range(self.cfg.optim_steps + self.cfg.extra_smooth_steps):
                start_time = time.time()
                if (
                    self.cfg.goal_set_proj
                    and alg_switch and t < self.cfg.optim_steps
                ):
                    self.learner.update_goal()
                    self.selected_goals.append(self.traj.goal_idx)

                self.info.append(self.optim.optimize(traj, force_update=True))
                self.history_trajectories.append(np.copy(traj.data))

                if self.cfg.report_time:
                    print("plan optimize:", time.time() - start_time)

                if self.info[-1]["terminate"] and t > 0:
                    break
                if self.cfg.timeout != -1 and time.time() - start_time_ > self.cfg.timeout and t > 0:
                    break

            # compute information for the final
            if not self.info[-1]["terminate"]:
                self.info.append(self.optim.optimize(traj, info_only=True))
            else:
                del self.history_trajectories[-1]
            plan_time = time.time() - start_time_

            res = (
                "SUCCESS BE GENTLE"
                if self.info[-1]["terminate"]
                else "FAIL DONT EXECUTE"
            )
            if not self.cfg.silent:
                print( "planning time: {:.3f} PLAN {} Length: {}".format(
                        plan_time, res, len(self.history_trajectories[-1])
                )
            )
            self.info[-1]["time"] = plan_time

        else:
            if not self.cfg.silent: print("planning not run...")
        return self.info
