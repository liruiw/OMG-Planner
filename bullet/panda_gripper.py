# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import os
import IPython

# adapted from https://github.com/bryandlee/franka_pybullet/tree/ac86319a0b2f6c863ba3c7ee3d52f4f51b2be3bd
class Panda:
    def __init__(
        self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0, 0, 0]
    ):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "torque"

        self.position_control_gain_p = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
        self.position_control_gain_d = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        f_max = 250
        self.max_torque = [
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
        ]

        # connect pybullet
        p.setRealTimeSimulation(self.realtime)

        # load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(current_dir + "/models")

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.robot = p.loadURDF(
            "panda/panda_gripper.urdf",
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self._base_position = [
            -0.05 - base_shift[0],
            0.0 - base_shift[1],
            -0.65 - base_shift[2],
        ]
        self.pandaUid = self.robot

        # robot parameters
        self.dof = p.getNumJoints(self.robot)
        c = p.createConstraint(
            self.robot,
            8,
            self.robot,
            9,
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []
        self.pandaEndEffectorIndex = 7

        for j in range(self.dof):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j]) / 2.0)
            self.target_torque.append(0.0)

        self.reset(init_joints)

    def reset(self, joints=None):
        self.t = 0.0
        self.control_mode = "torque"
        p.resetBasePositionAndOrientation(
            self.pandaUid, self._base_position, [0.000000, 0.000000, 0.000000, 1.000000]
        )
        if joints is None:
            self.target_pos = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0, 0.04, 0.04]
            for j in range(self.dof):
                self.target_torque[j] = 0.0
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j])
        else:
            joints = list(joints)
            joints.insert(7, 0)
            for j in range(self.dof):

                self.target_pos[j] = joints[j]
                self.target_torque[j] = 0.0
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j])

        self.resetController()
        self.setTargetPositions(self.target_pos)

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for i in range(self.dof)],
        )

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception("wrong control mode")

    def append(self, target_pos):
        if len(target_pos) == 9:
            if type(target_pos) == list:
                target_pos.insert(7, 0)
                return target_pos
            else:
                target_pos = np.insert(target_pos, 7, 0)
                return target_pos
        return target_pos

    def setTargetPositions(self, target_pos):
        self.target_pos = self.append(target_pos)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos,
            forces=self.max_torque,
            positionGains=self.position_control_gain_p,
            velocityGains=self.position_control_gain_d,
        )

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=self.target_torque,
        )

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def solveInverseDynamics(self, pos, vel, acc):
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))


if __name__ == "__main__":
    robot = Panda(realtime=1)
