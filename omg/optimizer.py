# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .util import *
import IPython


class Optimizer(object):
    """
    Optimizer class that runs optimization based on CHOMP
    """

    def __init__(self, scene, cost):
        self.cfg = scene.config
        self.joint_lower_limit = scene.robot.joint_lower_limit
        self.joint_upper_limit = scene.robot.joint_upper_limit
        self.cost = cost
        self.step = 0
        self.time = 0.0
        self.time_elapsed = time.time()

    def report(self, curve, info):
        """
        Record and print information for debug purpose
        """

        text = []
        if self.cfg.report_cost:
            text = [
                "=================================================",
                "step: {:.2f}, time: {:.2f}, lr: {:.5f}, collide: {}".format(
                    self.step, self.time_elapsed, self.cfg.step_size, info["collide"]
                ),
                "joint limit: {:.2f}/{:.2f}, {:.2f}/{:.2f}, violate: {} reach: {:.2f} timestep {}".format(
                    curve.min(),
                    self.joint_lower_limit.min(),
                    curve.max(),
                    self.joint_upper_limit.max(),
                    info["violate_limit"],
                    info["reach"],
                    self.cfg.timesteps,
                ),
                "obs:{:.2f}, smooth:{:.2f}, grasp:{:.2f} total:{:.2f} ".format(
                    info["obs"], info["smooth"], info["grasp"], info["cost"]
                ),
                "obs_grad:{:.2f}, smooth_grad:{:.2f}, grasp_grad:{:.2f} total_grad:{:.2f}".format(
                    info["weighted_obs_grad"],
                    info["weighted_smooth_grad"],
                    info["weighted_grasp_grad"],
                    info["grad"],
                ),
                "=================================================",
            ]
            for t in text:
                print(t)
        return text

    def update(self):
        """
        Update the state of the optimizer
        """
        self.step += 1
        self.time_elapsed = time.time() - self.time
        self.time = time.time()

        # cost schedule
        self.cfg.obstacle_weight = (
            self.cfg.base_obstacle_weight * self.cfg.cost_schedule_decay ** self.step
        )
        self.cfg.smoothness_weight = (
            self.cfg.smoothness_base_weight * self.cfg.cost_schedule_boost ** self.step
        )
        self.cfg.grasp_weight = (
            self.cfg.base_grasp_weight * self.cfg.cost_schedule_decay ** self.step
        )
        self.cfg.step_size = (
            self.cfg.step_decay_rate ** self.step * self.cfg.base_step_size
        )

    def reset(self):
        """
        Reset the state of the optimizer
        """
        self.step = 0

    def goal_set_projection(self, traj, grad):
        """
        Run one step of goal set optimization step
        """
        m, n = traj.end.shape[0], self.cfg.A.shape[0]
        if self.cfg.use_standoff:
            chosen_goal = self.cost.target_obj.reach_grasps[int(traj.goal_idx)]
            constraint_num = chosen_goal.shape[0]
        else:
            chosen_goal = traj.goal_set[int(traj.goal_idx)]
            constraint_num = 1
        cur_end_point = traj.data[-constraint_num:]

        target_goal_diff = cur_end_point - chosen_goal
        C = np.zeros([constraint_num, n])
        C[-constraint_num:, -constraint_num:] = np.eye(constraint_num)
        b = target_goal_diff

        # projected gradient step: unconstrained -> zero set projection -> offset
        M = self.cfg.Ainv.dot(C.T).dot(np.linalg.inv(C.dot(self.cfg.Ainv.dot(C.T))))
        update = (
            -self.cfg.step_size * self.cfg.Ainv.dot(grad)
            + self.cfg.step_size * M.dot(C).dot(self.cfg.Ainv).dot(grad)
            - M.dot(b)
        )
        return update

    def optimize(self, traj, force_update=False, info_only=False):
        """
        Run one step of chomp optimization
        """
        self.update()
        curve = traj.data
        cost, grad, info = self.cost.compute_total_loss(traj)
        self.check_joint_limit(curve, info)
        info["text"] = self.report(curve, info)
        if (info["terminate"] and not force_update) or info_only:
            return info

        if self.cfg.goal_set_proj:
            update = self.goal_set_projection(traj, grad)
            traj.update(update)
            traj.set(self.handle_joint_limit(traj.data))
        else:
            update = -self.cfg.step_size * self.cfg.Ainv.dot(grad)
            traj.update(update)
            traj.set(self.handle_joint_limit(traj.data))
        return info

    def compute_traj_v(self, curve):
        """
        Compute L1 projection to the joint limit.
        """
        low_mask = curve < self.joint_lower_limit
        high_mask = curve > self.joint_upper_limit
        traj_v = low_mask * (self.joint_lower_limit - curve) + high_mask * (
            self.joint_upper_limit - curve
        )
        return traj_v

    def handle_joint_limit(self, curve):
        """
        Make the hard L1 projection smooth.
        """
        cnt = 0
        traj_v = self.compute_traj_v(curve)

        while (np.linalg.norm(traj_v) > 1e-2) and cnt < self.cfg.joint_limit_max_steps:
            traj_vstar = self.cfg.Ainv.dot(traj_v)
            maxidx = np.unravel_index(np.abs(traj_v).argmax(), traj_v.shape)

            scale = safe_div(np.abs(traj_v).max(), (np.abs(traj_vstar[maxidx])))
            curve = curve + scale * traj_vstar
            traj_v = self.compute_traj_v(curve)
            cnt += 1

        return curve

    def check_joint_limit(self, curve, info):
        """
        Check joint limit violation
        """
        low_mask = (curve < self.joint_lower_limit - 5e-3).any()
        high_mask = curve > self.joint_upper_limit + 5e-3
        over_joint_limit = (low_mask * high_mask).any()  #
        info["violate_limit"] = over_joint_limit
        info["terminate"] = info["terminate"] and (not over_joint_limit)
