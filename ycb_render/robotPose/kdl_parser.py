# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import PyKDL as kdl
from .urdf_parser_py.urdf import URDF
import random


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
    cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def urdf_pose_to_kdl_frame(pose):
    pos = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position

        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)), kdl.Vector(*pos))


def urdf_joint_to_kdl_joint(jnt):

    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    try:  # kdl issue to run with python 2
        fixed_joint = kdl.Joint.Fixed
    except:
        joint_type_str = "kdl.Joint.None"
        fixed_joint = eval(joint_type_str)

    if jnt.joint_type == "fixed":
        return kdl.Joint(jnt.name, fixed_joint)  # None
    axis = kdl.Vector(*[float(s) for s in jnt.axis])
    if jnt.joint_type == "revolute":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "continuous":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "prismatic":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.TransAxis
        )
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name, fixed_joint)


def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(
        i.mass,
        origin.p,
        kdl.RotationalInertia(
            i.inertia.ixx,
            i.inertia.iyy,
            i.inertia.izz,
            i.inertia.ixy,
            i.inertia.ixz,
            i.inertia.iyz,
        ),
    )
    return origin.M * rbi


# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)

    ef_tree = dict()  # store the names for now

    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            if len(urdf.child_map[parent]) > 1:
                ef_tree[parent] = [
                    child_name for _, child_name in urdf.child_map[parent]
                ]
            for joint, child_name in urdf.child_map[parent]:
                for lidx, link in enumerate(urdf.links):
                    if child_name == link.name:
                        child = urdf.links[lidx]
                        if child.inertial is not None:
                            kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                        else:
                            kdl_inert = kdl.RigidBodyInertia()
                        for jidx, jnt in enumerate(urdf.joints):
                            if jnt.name == joint:
                                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joints[jidx])
                                kdl_origin = urdf_pose_to_kdl_frame(
                                    urdf.joints[jidx].origin
                                )
                                kdl_sgm = kdl.Segment(
                                    child_name, kdl_jnt, kdl_origin, kdl_inert
                                )
                                tree.addSegment(kdl_sgm, parent)
                                add_children_to_tree(child_name)

    add_children_to_tree(root)
    return tree, ef_tree


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(usage="Load an URDF file")
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        nargs="?",
        default="baxter_base.urdf",
        help="The URDF file",
    )
    args = parser.parse_args()
    robot = URDF.from_xml_string(
        args.file.read()
    )  # urdf.Robot.from_xml_file(sys.argv[1])
    num_non_fixed_joints = 0

    for joint in robot.joints:
        if joint.limit:
            print(
                "joint name: {}, limit: ({}, {}), parent :{}, child: {}".format(
                    joint.name,
                    joint.limit.lower,
                    joint.limit.upper,
                    joint.parent,
                    joint.child,
                )
            )
        if joint.joint_type != "fixed":
            num_non_fixed_joints += 1
        else:
            print(
                "fixed joint name: {}, parent :{}, child: {}".format(
                    joint.name, joint.parent, joint.child
                )
            )
    tree, ef = kdl_tree_from_urdf_model(robot)
    print("URDF non-fixed joints: %d;" % num_non_fixed_joints)
    print("KDL joints: %d" % tree.getNrOfJoints())
    print("Segments: %d" % tree.getNrOfSegments())

    base_link = robot.get_root()
    end_link = robot.link_map.keys()[len(robot.links) - 1]
    chain = tree.getChain(base_link, end_link)

    for idx in xrange(chain.getNrOfSegments()):
        segment = chain.getSegment(idx)
        print(segment.pose(0))  # fix joint won't change with joint angle


if __name__ == "__main__":
    main()
