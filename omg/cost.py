# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .util import *
import time
import torch
import IPython
from layers.sdf_matching_loss import SDFLoss


class Cost(object):
    """
    Cost class that computes obstacle, grasp, and smoothness and their gradients.
    """

    def __init__(self, env):
        self.env = env
        self.cfg = env.config
        self.sdf_loss = SDFLoss()
        if len(self.env.objects) > 0:
            self.target_obj = self.env.objects[self.env.target_idx]

    def functional_grad(self, v, a, JT, ws_cost, ws_grad):
        """
        Compute functional gradient based on workspace cost.
        """
        p = v.shape[-2]
        vel_norm = np.linalg.norm(v, axis=-1, keepdims=True)  # n x p x 1
        cost = np.sum(ws_cost * vel_norm[..., 0], axis=-1)  # n
        normalized_vel = safe_div(v, vel_norm)  # p x 3
        proj_mat = np.eye(3) - np.matmul(
            normalized_vel[..., None], normalized_vel[..., None, :]
        )  # p x 3 x 3
        scaled_curvature = ws_cost[..., None, None] * safe_div(
            np.matmul(proj_mat, a[..., None]), vel_norm[..., None] ** 2
        )  # p x 3 x 1
        projected_grad = np.matmul(proj_mat, ws_grad[..., None])  # p x 3 x 1
        grad = np.sum(
            np.matmul(JT, (vel_norm[..., None] * projected_grad - scaled_curvature)),
            axis=-1,
        )
        return cost, grad

    def forward_poses(self, joints):
        """
        Compute forward kinematics for link poses and joint origins and axis
        """
        robot = self.env.robot
        (
            robot_poses,
            joint_origins,
            joint_axis,
        ) = robot.robot_kinematics.forward_kinematics_parallel(
            joints[None, ...], base_link=self.cfg.base_link, return_joint_info=True
        )

        return robot_poses[0], joint_origins[0], joint_axis[0]

    def forward_points(self, pose, pts, normals=None):
        """
        Map the points through forward kinematics.
        """
        r = pose[..., :3, :3]
        t = pose[..., :3, [3]]
        x = np.matmul(r, pts[None, ...]) + t
        if normals is None:
            return x.transpose([3, 1, 0, 2])  # p x n x m x 3
        else:
            normal = np.matmul(r, normals[None, ...])
            x = np.concatenate([x, normal], 2)
            return x.transpose([3, 1, 0, 2])

    def color_point(self, vis_pts, collide):
        """
        Color the sampled points based on relative sdf value.
        Green is closed to obstacle, blue is far, yellow is gradient direction
        """
        p_max, p_min = (
            np.amax(vis_pts[..., 6], axis=(-2, -1))[..., None, None],
            np.amin(vis_pts[..., 6], axis=(-2, -1))[..., None, None],
        )
        vis_pts[..., 6] = 255 * safe_div(
            vis_pts[..., 6] - p_min, (p_max - p_min + 1e-8)
        )
        vis_pts[..., 7] = 255 - vis_pts[..., 6]
        if type(collide) is torch.Tensor:
            collide = collide.detach().cpu().numpy()
        collide = collide.astype(np.bool)
        vis_pts[collide, 6:9] = 255, 0, 0

    def compute_point_jacobian(
        self, joint_origin, x, joint_axis, potentials, type="revolute"
    ):
        """
        Compute jacobian transpose for each point on a link.
        """
        p, n = x.shape[:2]
        x = x.transpose([1, 0, 2])[:, :, None, :]
        m = joint_axis.shape[1]
        jacobian = np.zeros([n, p, m, 6])
        jacobian[..., :3] = np.cross(
            joint_axis[:, None, ...], x - joint_origin[:, None, ...]
        )
        jacobian[..., 3:] = joint_axis[:, None, ...]
        if type == "prsimatic":  # finger joint
            jacobian[..., -1, :3] = joint_axis[:, [-1], :]
            jacobian[..., -1, 3:] = 0

        return jacobian

    def forward_kinematics_obstacle(self, xi, start, end, arc_length=True):
        """
        Instead of computing C space velocity and using Jacobian,
        we differentiate workspace positions for velocity and acceleration.
        """
        robot = self.env.robot
        robot_pts = robot.collision_points.transpose([0, 2, 1])  # m x 3 x p
        n, m = xi.shape[0], xi.shape[1]
        p = robot_pts.shape[2]
        vis_pts = np.zeros([n, m + 1, p, 12])
        Js = []  # (m + 1) x n x p x j x 6

        (
            robot_poses,
            joint_origins,
            joint_axis,
        ) = self.env.robot.robot_kinematics.forward_kinematics_parallel(
            wrap_values(xi), return_joint_info=True
        )
        ws_positions = self.forward_points(
            robot_poses, robot_pts
        )  # p x (m + 1) x n x 3

        potentials, potential_grads, collide = self.compute_obstacle_cost_layer(
            torch.from_numpy(ws_positions).cuda().float().permute(2, 1, 0, 3),
            vis_pts,
            uncheck_finger_collision=self.cfg.uncheck_finger_collision,
            special_check_id=self.env.target_idx,
        )
        potentials = potentials.detach().cpu().numpy()
        potential_grads = potential_grads.detach().cpu().numpy()
        collide = collide.detach().cpu().numpy()
        self.color_point(vis_pts, collide)

        """ compute per point jacobian """
        for j in range(m + 1):
            Js.append(
                self.compute_point_jacobian(
                    joint_origins[:, wrap_joint(j + 1)],
                    ws_positions[:, j],
                    joint_axis[:, wrap_joint(j + 1)],
                    potentials[:, j],
                    "prsimatic" if j >= 8 else "revolute",
                )
            )
        """ endpoint """
        if arc_length:
            robot_poses_start = self.forward_poses(wrap_value(start))[0]
            robot_poses_end = self.forward_poses(wrap_value(end))[0]

            ws_positions_start = self.forward_points(
                robot_poses_start[None, ...], robot_pts)[:, :, 0]
            ws_positions_end = self.forward_points(
                robot_poses_end[None, ...],   robot_pts)[:, :, 0]

            """ get derivative """
            ws_velocity = self.cfg.get_derivative(
                ws_positions, ws_positions_start, ws_positions_end, 1
            )
            ws_acceleration = self.cfg.get_derivative(
                ws_positions, ws_positions_start, ws_positions_end, 2
            )
            ws_velocity = ws_velocity.transpose([2, 1, 0, 3])
            ws_acceleration = ws_acceleration.transpose(
                [2, 1, 0, 3]
            )  # n x (m + 1) x p x 3
            ws_positions = ws_positions.transpose([2, 1, 0, 3])
            return (
                ws_positions,
                ws_velocity,
                ws_acceleration,
                Js,
                potentials,
                potential_grads,
                vis_pts,
                collide.sum(),
            )
        else:
            return Js, potentials, potential_grads, collide.sum()

    def batch_obstacle_cost(
        self,
        joints,
        arc_length=-1,
        only_collide=False,
        special_check_id=0,
        uncheck_finger_collision=-1,
        start=None,
        end=None,
    ):
        """
        Compute obstacle cost given a batch of joints
        """
        s = time.time()
        robot_pts = self.env.robot.collision_points.transpose([0, 2, 1])
        robot_poses = []

        robot_poses = self.env.robot.robot_kinematics.forward_kinematics_parallel(
            wrap_values(joints)
        )
        if self.cfg.report_time:
            print("fk time:", time.time() - s)

        ws_positions = self.forward_points(
            robot_poses, robot_pts
        )  # p x (m + 1) x n x 3
        ws_positions = torch.from_numpy(ws_positions).cuda().float()
        vis_pts = np.zeros(
            [ws_positions.shape[2], ws_positions.shape[1], ws_positions.shape[0], 12]
        )
        collision_check_func = self.compute_obstacle_cost_layer
        s = time.time()
        potentials, grad, collide = collision_check_func(
            ws_positions.permute(2, 1, 0, 3),
            vis_pts=vis_pts,
            special_check_id=special_check_id,
            uncheck_finger_collision=uncheck_finger_collision,
        )
        self.color_point(vis_pts, collide)
        if self.cfg.report_time:
            print("obstacle time:", time.time() - s)
        s = time.time()

        if arc_length > 0:  # p x m x g x n x 3
            ws_positions = ws_positions.reshape(
                ws_positions.shape[0], ws_positions.shape[1], -1, arc_length, 3
            )
            ws_positions = ws_positions.permute(2, 1, 0, 3, 4)  # g x m x p x n x 3
            robot_poses_start = self.forward_poses(wrap_value(start))[0]
            robot_poses_end = (
                self.env.robot.robot_kinematics.forward_kinematics_parallel(
                    wrap_values(end)
                )
            )
            robot_poses_start = np.tile(
                robot_poses_start, (robot_poses_end.shape[0], 1, 1, 1)
            )
            ws_positions_start = self.forward_points(
                robot_poses_start, robot_pts
            )  # p x m x g x 3
            ws_positions_end = self.forward_points(robot_poses_end, robot_pts)
            ws_positions_start = (
                torch.from_numpy(ws_positions_start).cuda().float().permute(2, 1, 0, 3)
            )
            ws_positions_end = (
                torch.from_numpy(ws_positions_end).cuda().float().permute(2, 1, 0, 3)
            )

            ws_velocities = self.cfg.get_derivative_torch(
                ws_positions, ws_positions_start, ws_positions_end
            )
            ws_velocities = ws_velocities.transpose(1, 3)  # g x n x p x m x 3
            ws_velocities = torch.norm(
                ws_velocities.reshape(
                    [
                        -1,
                        ws_velocities.shape[2],
                        ws_velocities.shape[3],
                        ws_velocities.shape[4],
                    ]
                ),
                dim=-1,
            ).transpose(-1, -2)
            potentials = potentials * ws_velocities  # (g x n) x m x p

        if self.cfg.report_time:
            print("arc length time:", time.time() - s)
        if only_collide:
            collide_mask = (
                potentials
                > 0.5 * (self.cfg.epsilon - self.cfg.clearance) ** 2 / self.cfg.epsilon
            ).any()
            potentials = potentials * collide_mask

        return potentials, grad, vis_pts, collide

    def compute_obstacle_cost_layer(
        self,
        ws_positions,
        vis_pts=None,
        special_check_id=0,
        uncheck_finger_collision=-1,
        grad_free=True,
    ):
        """
        Compute obstacle cost and gradient from sdf, take in torch cuda tensor
        """
        # prepare data
        n, m, p, _ = ws_positions.shape
        points = ws_positions.reshape([-1, 3])

        num_objects = len(self.env.objects)
        poses = np.zeros((num_objects, 4, 4), dtype=np.float32)
        epsilons = np.zeros((num_objects,), dtype=np.float32)
        padding_scales = np.zeros((num_objects,), dtype=np.float32)
        clearances = np.zeros((num_objects,), dtype=np.float32)
        disables = np.zeros((num_objects,), dtype=np.float32)

        for idx, obs in enumerate(self.env.objects):
            if obs.name == "floor" or obs.name in self.cfg.disable_collision_set:
                disables[idx] = 1
 
            padding_scale = 1
            eps = self.cfg.epsilon
            clearances[idx] = self.cfg.clearance
            epsilons[idx] = eps
            padding_scales[idx] = padding_scale
            poses[idx] = se3_inverse(obs.pose_mat)

        # forward layer
        poses = torch.from_numpy(poses).cuda()
        epsilons = torch.from_numpy(epsilons).cuda()
        clearances = torch.from_numpy(clearances).cuda()
        padding_scales = torch.from_numpy(padding_scales).cuda()
        disables = torch.from_numpy(disables).cuda()
        potentials, potential_grads, collides = self.sdf_loss(
            poses,
            self.env.sdf_torch,
            self.env.sdf_limits,
            points,
            epsilons,
            padding_scales,
            clearances,
            disables,
        )
        potentials = potentials.reshape([n, m, p])
        potential_grads = potential_grads.reshape([n, m, p, 3])
        collides = collides.reshape([n, m, p])

        if self.cfg.use_standoff and self.cfg.goal_set_proj:
            potentials[-self.cfg.reach_tail_length :] = 0
            potential_grads[-self.cfg.reach_tail_length :] = 0
            collides[-self.cfg.reach_tail_length :] = 0

        if uncheck_finger_collision == -1:
            potentials[:, -2:] *= 0.1  # soft
            potential_grads[:, -2:] *= 0.1  # soft
            collides[:, -2:] = 0

        if vis_pts is not None:
            vis_pts[:, :m, :, :3] = points.reshape([n, m, p, 3]).detach().cpu().numpy()
            vis_pts[:, :m:, :, 6] = potentials.detach().cpu().numpy()
            vis_pts[:, :m:, :, 9:] = potential_grads.detach().cpu().numpy()

        return potentials, potential_grads, collides

    def compute_collision_loss(self, xi, start, end):
        """
        Computes obstacle loss
        """
        n, m = xi.shape[0], xi.shape[1]
        obs_grad = np.zeros_like(xi)
        obs_cost = np.zeros([n, m + 1])
        (
            x,
            v,
            a,
            Js,
            potentials,
            potential_grads,
            vis_pts,
            collide,
        ) = self.forward_kinematics_obstacle(xi, start, end)

        if self.cfg.top_k_collision == 0:
            for j in range(m + 1):
                J = np.array(Js[j])[..., :3]
                obs_cost_i, obs_grad_i = self.functional_grad(
                    v[:, j], a[:, j], J, potentials[:, j], potential_grads[:, j]
                )
                obs_grad_i = obs_grad_i.sum(1)
                obs_cost[:, j] += obs_cost_i
                obs_grad[:, wrap_index(j + 1)] += obs_grad_i.reshape([n, -1])

        else:
            # top k collision points in the whole trajectory
            topk = np.unravel_index(np.argsort(potentials.flatten()), potentials.shape)
            top_n, top_m, top_p = (
                topk[0][-self.cfg.top_k_collision :],
                topk[1][-self.cfg.top_k_collision :],
                topk[2][-self.cfg.top_k_collision :],
            )
            top_potentials = potentials[top_n, top_m, top_p]
            vis_pts[top_n, top_m, top_p, 6:9] = [235, 52, 195]

            if not self.cfg.consider_finger:
                m = m - 2

            for j in range(m + 1):
                if np.isin(j, top_m):
                    mask = j == top_m
                    select_n, select_p = top_n[mask], top_p[mask]
                    J = np.array(Js[j])[select_n, select_p, :, :3]
                    obs_cost_i, obs_grad_i = self.functional_grad(
                        v[select_n, j, select_p],
                        a[select_n, j, select_p],
                        J,
                        potentials[select_n, j, select_p],
                        potential_grads[select_n, j, select_p],
                    )
                    obs_cost[:, j] += obs_cost_i
                    select_m = wrap_index(j + 1)
                    select_n, select_m = np.repeat(select_n, len(select_m)), np.tile(
                        select_m, len(select_n)
                    )
                    obs_grad[select_n, select_m] += obs_grad_i.flatten()

        return obs_cost, obs_grad, vis_pts, collide.sum()

    def compute_smooth_loss(self, xi, start, end):
        """
        Computes smoothness loss
        """
        link_smooth_weight = np.array(self.cfg.link_smooth_weight)[None]
        ed = np.zeros([xi.shape[0] + 1, xi.shape[1]])
        ed[0] = (
            self.cfg.diff_rule[0][self.cfg.diff_rule_length // 2 - 1]
            * start
            / self.cfg.time_interval
        )
        if not self.cfg.goal_set_proj:
            ed[-1] = (
                self.cfg.diff_rule[0][self.cfg.diff_rule_length // 2]
                * end
                / self.cfg.time_interval
            )

        velocity = self.cfg.diff_matrices[0].dot(xi)  #
        velocity_norm = np.linalg.norm((velocity + ed) * link_smooth_weight, axis=1)
        smoothness_loss = 0.5 * velocity_norm ** 2  #

        smoothness_grad = self.cfg.A.dot(xi) + self.cfg.diff_matrices[0].T.dot(ed)
        smoothness_grad *= link_smooth_weight
        return smoothness_loss, smoothness_grad

    def compute_total_loss(self, traj):
        """
        Compute total losses, gradients, and other info
        """
        smoothness_loss, smoothness_grad = self.compute_smooth_loss(
            traj.data, traj.start, traj.end
        )
        (
            obstacle_loss,
            obstacle_grad,
            collision_pts,
            collide,
        ) = self.compute_collision_loss(traj.data, traj.start, traj.end)
        smoothness_loss_sum = smoothness_loss.sum()
        obstacle_loss_sum = obstacle_loss.sum()

        weighted_obs = self.cfg.obstacle_weight * obstacle_loss_sum
        weighted_smooth = self.cfg.smoothness_weight * smoothness_loss_sum
        weighted_obs_grad = self.cfg.obstacle_weight * obstacle_grad
        weighted_obs_grad = np.clip(
            weighted_obs_grad, -self.cfg.clip_grad_scale, self.cfg.clip_grad_scale
        )
        weighted_smooth_grad = self.cfg.smoothness_weight * smoothness_grad

        cost = weighted_obs + weighted_smooth
        grad = weighted_obs_grad + weighted_smooth_grad

        cost_traj = (
            self.cfg.obstacle_weight * obstacle_loss.sum(-1)
            + self.cfg.smoothness_weight * smoothness_loss[:-1]
        )
   
        goal_dist = (
            np.linalg.norm(traj.data[-1] - traj.goal_set[traj.goal_idx])
            if self.cfg.goal_set_proj
            else 0
        )
         
        terminate = (
            (collide <= self.cfg.allow_collision_point)
            and self.cfg.pre_terminate
            and (goal_dist < 0.01)
            and smoothness_loss_sum < self.cfg.terminate_smooth_loss
        )
         
        execute = (collide <= self.cfg.allow_collision_point) and (
            smoothness_loss_sum < self.cfg.terminate_smooth_loss
        )
        standoff_idx = (
            len(traj.data) - self.cfg.reach_tail_length
            if self.cfg.use_standoff
            else len(traj.data) - 1
        )
        info = {
            "collision_pts": collision_pts,
            "obs": obstacle_loss_sum,
            "smooth": smoothness_loss_sum,
            "grasp": 0,
            "weighted_obs": weighted_obs,
            "weighted_smooth": weighted_smooth,
            "weighted_smooth_grad": np.linalg.norm(weighted_smooth_grad),
            "weighted_obs_grad": np.linalg.norm(weighted_obs_grad),
            "weighted_grasp_grad": 0,
            "weighted_grasp": 0,
            "gradient": grad,
            "cost": cost,
            "grad": np.linalg.norm(grad),
            "terminate": terminate,
            "collide": collide,
            "standoff_idx": standoff_idx,
            "reach": goal_dist,
            "execute": execute,
            "cost_traj": cost_traj,
        }

        return cost, grad, info
