# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import sys
from . import _init_paths
from omg.util import *
from omg.core import *
from omg import config
import time
import argparse
import IPython
import scipy.io as sio
import platform

PYTHON2 = True
if platform.python_version().startswith("3"):
    input_func = input
else:
    input_func = raw_input


def plan_to_target(scene, start_conf, target_name):
    """
    plan a grasp trajectory for a target object
    """
    scene.traj.start = start_conf
    scene.env.set_target(target_name)
    scene.reset(lazy=True)
    info = scene.step()
    joint_trajectory_points = np.concatenate(
        [scene.planner.history_trajectories[-1]], axis=0
    )
    return joint_trajectory_points


def plan_to_conf(
    scene,
    start_conf,
    end_conf,
    disable_list=[
        "kitchen_front_vertical_right",
        "kitchen_sektion_top_three",
        "kitchen_handle",
    ],
):
    """
    plan a trajectory between two fixed configurations
    """
    scene.traj.start = start_conf
    dummy_goal_set = config.cfg.goal_set_proj
    config.cfg.disable_collision_set = disable_list  # add handle and top drawer here
    setattr(config.cfg, "goal_set_proj", False)
    dummy_standoff = config.cfg.use_standoff
    config.cfg.use_standoff = False
    config.cfg.get_global_param()
    scene.traj.end = end_conf
    scene.reset(lazy=True)
    info = scene.step()
    joint_trajectory_points = np.concatenate([scene.traj.data])
    setattr(config.cfg, "goal_set_proj", dummy_goal_set)
    setattr(config.cfg, "use_standoff", dummy_standoff)
    config.cfg.disable_collision_set = []
    config.cfg.get_global_param()
    return joint_trajectory_points


def place_target(
    scene,
    start_conf,
    target_name,
    place_translation=[0.0, -0.3, 0],
    apply_standoff=False,
    write_video=False,
    vis_once=False,
    vis_collision=False,
    not_vis=False,
):
    """
    place a placement trajectory for a target object to a delta translation
    """
    scene.traj.start = start_conf
    config.cfg.disable_collision_set = [target_name]
    dummy_standoff = config.cfg.use_standoff

    config.cfg.use_standoff = apply_standoff
    if apply_standoff:
        config.cfg.increment_iks = True
    obj_idx = scene.env.names.index(target_name)

    # compute relative pose and attach object
    target = scene.env.objects[obj_idx]
    grasp_pose = target.pose.copy()
    start_joints = wrap_value(start_conf)
    robot = scene.env.robot

    # attach
    start_hand_pose = robot.robot_kinematics.forward_kinematics_parallel(
        start_joints[None, ...], base_link=config.cfg.base_link
    )[0][7]

    target.rel_hand_pose = relative_pose(
        pack_pose(start_hand_pose), grasp_pose
    )  # object -> hand
    place_pose = grasp_pose.copy()
    place_pose[:3] += np.array(place_translation)  #
    scene.env.update_pose(target_name, place_pose)
    vis = (
        input_func("visualize place pose? (y/n)")
        if not vis_once and hasattr(scene, "renderer")
        else "f"
    )

    if vis == "y" and hasattr(scene, "renderer"):
        scene.fast_debug_vis(
            interact=2, collision_pt=vis_collision, traj_type=0, write_video=write_video
        )

    target.attached = True
    scene.env.set_target(target_name)
    scene.reset(lazy=True)
    robot.resample_attached_object_collision_points(target)

    if len(scene.traj.goal_set) == 0:  # placement fail
        print("please update place pose, there is no IK")
        config.cfg.disable_collision_set = []
        target.attached = False
        setattr(config.cfg, "use_standoff", dummy_standoff)
        scene.env.update_pose(target_name, grasp_pose)
        robot.reset_hand_points()
        return None

    else:
        info = scene.step()
        joint_trajectory_points = np.concatenate(
            [scene.planner.history_trajectories[-1]], axis=0
        )
        config.cfg.disable_collision_set = []
        standoff_idx = info[-1]["standoff_idx"]
        setattr(config.cfg, "use_standoff", dummy_standoff)
        end_hand_pose = robot.robot_kinematics.forward_kinematics_parallel(
            wrap_value(joint_trajectory_points[[standoff_idx], ...]),
            base_link=config.cfg.base_link,
        )[0][7]

        place_pose = compose_pose(pack_pose(end_hand_pose), target.rel_hand_pose)
        scene.env.update_pose(target_name, place_pose)  # update delta pose
        vis = (
            input_func("visualize? (y/n)")
            if not vis_once and hasattr(scene, "renderer")
            else "y"
        )

        while vis == "y" and hasattr(scene, "renderer"):
            if apply_standoff:
                traj_data = scene.planner.history_trajectories[-1]
                scene.fast_debug_vis(
                    traj=traj_data[:standoff_idx],
                    interact=1,
                    collision_pt=vis_collision,
                    nonstop=vis_once,
                    write_video=write_video,
                )
                target.attached = False  # open the finger here
                scene.fast_debug_vis(
                    traj=traj_data[standoff_idx:],
                    interact=1,
                    collision_pt=vis_collision,
                    nonstop=vis_once,
                    traj_type=1,
                    write_video=write_video,
                )
            else:
                scene.fast_debug_vis(
                    interact=1,
                    collision_pt=vis_collision,
                    nonstop=vis_once,
                    write_video=write_video,
                )
            if vis_once:
                break
            vis = input_func("visualize? (y/n)")

        target.attached = False
        robot.reset_hand_points()
        return joint_trajectory_points


def populate_scene(
    scene, object_list, object_poses, flag_compute_grasp, add_table=False
):
    """
    load the objects
    """
    for i, name in enumerate(object_list):
        scene.env.add_object(
            name,
            object_poses[i][:3],
            object_poses[i][3:],
            compute_grasp=flag_compute_grasp[i],
        )
    if add_table:
        scene.env.add_plane(object_poses[-2][:3], object_poses[-2][3:])
        scene.env.add_table(object_poses[-1][:3], object_poses[-1][3:])
    scene.env.combine_sdfs()


def interface(
    scene, start_conf, object_list, object_poses, flag_compute_grasp, add_table=False
):
    """
    populate the objects for setting up the scene
    """
    scene.traj.start = start_conf
    scene.env.clear()
    start_time = time.time()
    populate_scene(
        scene, object_list, object_poses, flag_compute_grasp, add_table=False
    )
    print("populate scene time: {:.3f}".format(time.time() - start_time))


def print_env_info(scene, env, end_conf):
    """
    print out current objects poses and configuration
    """
    print("====================================================")
    print("start configuration: {}".format(scene.traj.start.flatten()))
    print("end configuration: {}".format(end_conf.flatten()))
    for obj in env.objects:
        if obj.name.startswith("0"):
            print("object name: {} pose: {}".format(obj.name, obj.pose.flatten()))
    print("====================================================")


def main():
    """
    Act as an example for usage of OMG in pick and place and is not fully tested.
    """
    np.set_printoptions(2)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--init", help="trajectory initialization", type=str, default="grasp"
    )
    parser.add_argument("-v", "--vis", help="visualization", action="store_true")
    parser.add_argument(
        "-vc", "--vis_collision", help="visualization", action="store_true"
    )
    parser.add_argument("-f", "--file", help="load from mat file", default="kitchen0")
    parser.add_argument(
        "-s", "--script", help="load from mat file", default="script.txt"
    )
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")

    args = parser.parse_args()
    place_script_idx, place_poses = 0, []
    target_script_idx, target_names = 0, []
    end_script_idx, end_confs = 0, []
    target = None

    vis_once = False
    vis_collision = args.vis_collision
    write_video = args.write_video
     
    config.cfg.goal_set_max_num = 100
    mkdir_if_missing('output_videos')
    config.cfg.output_video_name = "output_videos/" + args.file + ".avi"
    config.cfg.report_cost = False
    args.file = args.file + ".mat"
    file_name = config.cfg.root_dir + "real_world/" + args.script
    
    if args.script is not None and os.path.exists(file_name):
        text_file = open( file_name, "r" )  # 'script.txt'
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

    config.cfg.traj_init  = "grasp"
    config.cfg.scene_file = ""
    config.cfg.grasp_optimize = False

    scene = PlanningScene(config.cfg)
    start_conf = np.array([0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    obj_pose = []

    if args.file is not None:  # kitchen
        start_conf = np.array(
            [-0.05929, -1.6124, -0.197, -2.53, -0.0952, 1.678, 0.587, 0.0398, 0.0398]
        )
        mat = sio.loadmat(config.cfg.root_dir + "data/scenes/" + args.file)
        object_lists = [f.strip() for f in mat["object_lists"]]  # .encode("utf-8")
        file_num = len(
            [
                name
                for name in object_lists
                if name.startswith("0") or name.startswith("kitchen")
            ]
        )
        files = range(file_num)
        file_num = len(files)
        object_lists = [object_lists[i] for i in files]  #
        config.cfg.cam_V = np.array(
            [
                [0.54, -0.78, -0.33, -0.22],
                [-0.57, -0.05, -0.82, 0.41],
                [0.63, 0.63, -0.46, 1.36],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        if "joint_position" in mat:
            start_conf = mat["joint_position"][0]
            config.cfg.cam_V = np.array(
                [
                    [-0.39, -0.92, 0.0, 0.44],
                    [-0.57, 0.24, -0.78, 0.64],
                    [0.72, -0.31, -0.62, 2.55],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        obj_pose = mat["object_poses"][files]
        print(object_lists)

    else:
        object_lists = ["021_bleach_cleanser"]
        obj_pose.append(np.array(([0.61, -0.027, 0.161, 0, 0, 0, 1])))

        obj_pose.append(np.array(([0.5, 0.0, -0.17, 0.0, 0.0, 0, 1])))
        obj_pose.append(np.array(([0.5, 0.0, -0.17, 0.0, 0.0, 0.707, 0.707])))

    end_conf = np.array(
        [-0.05929, -1.6124, -0.197, -2.53, -0.0952, 1.678, 0.587, 0.0398, 0.0398]
    )
    flag_compute_grasp = []
    for name in object_lists:
        if name.startswith("0"):
            flag_compute_grasp.append(True)
        else:
            flag_compute_grasp.append(False)
    interface(
        scene,
        start_conf,
        object_lists,
        obj_pose,
        flag_compute_grasp,
        args.file is not None,
    )  #
    scene.reset()

    while True:
        print_env_info(scene, scene.env, end_conf)

        # grasp
        if len(target_names) == 0:
            target_ = input_func("enter target name or quit:  ")
        elif target_script_idx < len(target_names):
            target_ = target_names[target_script_idx]
        else:
            target_ = ""
        if target_ == "quit":
            break

        if target_ in scene.env.names:
            target = target_
            traj = plan_to_target(scene, start_conf, target)
            vis = input_func("visualize? (y/n)") if not vis_once else "y"
            while vis == "y":
                scene.fast_debug_vis(
                    interact=1,
                    collision_pt=vis_collision,
                    nonstop=vis_once,
                    write_video=write_video,
                )
                if vis_once:
                    break
                vis = input_func("visualize? (y/n)")
        else:
            print("Set target example: 004_sugar_box")

        # update objects
        object_pose_ = (
            input_func("update object name and pose:  ").split(" ")
            if args.script is None
            else []
        )
        if len(object_pose_) == 2:
            object_pose = object_pose_
            obj, pose = object_pose
            pose = np.array([float(item) for item in pose.split(",")])
            if obj in scene.env.names and len(pose) == 7:
                scene.env.update_pose(obj, pose)
        else:
            print("Set target_pose example: 004_sugar_box 0.50,-0.07,0.14,1,0,0,0")

        object_pose_ = (
            input_func("add object name and pose:  ").split(" ")
            if args.script is None
            else []
        )
        if len(object_pose_) == 2:
            object_pose = object_pose_
            obj, pose = object_pose
            pose = np.array([float(item) for item in pose.split(",")])
            if len(pose) == 7:
                scene.insert_object(obj, pose[:3], pose[3:])
        else:
            print("Add object example: 019_pitcher_base 0.448,0.119,0.196,0,0,0,1")

        obj = input_func("remove object name: ") if args.script is None else ""
        if obj in scene.env.names:
            scene.env.remove_object(obj)
        else:
            print("Remove object example: 019_pitcher_base")

        # update configuration
        assign = input_func("Update config: ").split("=") if args.script is None else []
        if len(assign) == 2:
            attr, val = assign
            setattr(config.cfg, attr, float(val))
        else:
            print("Update config example: target_obj_collision=0.5")

        # placement
        if len(place_poses) == 0:
            place_pose_ = input_func("Place target delta translation: ").split(",")
        elif place_script_idx < len(place_poses):
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
                vis_collision=args.vis_collision,
                vis_once=vis_once,
                write_video=write_video,
            )
            if traj_ is not None:
                traj = traj_
        else:
            print("Place example: -0.45,0,0,0")

        # update start configuration
        start_conf_ = (
            input_func("update start configuration:  ").split(",")
            if args.script is None
            else []
        )
        if len(start_conf_) == 9:
            start_conf = np.array([float(item) for item in start_conf_])
        else:
            print(
                "Set start_config example: 0.5,-1.285,0.0,-2.356,0.0,1.571,0.785,0.04,0.04"
            )

        # update end configuration
        if len(end_confs) == 0:
            end_config = input_func("plan to fix end configuration:  ").split(",")
        elif target_script_idx < len(end_confs):
            end_config = end_confs[end_script_idx]
        else:
            end_config = ""
        if len(end_config) == 9:
            end_conf = np.array([float(item) for item in end_config])
            start_conf_test = start_conf
            if len(start_conf_test) != 9:
                start_conf_test = np.array(
                    [1.690, -1.562, -1.504, -1.93, -1.63, 1.57, 2.50, 0.04, 0.04]
                )  #
            plan_to_conf(
                scene, start_conf_test, end_conf, disable_list=["004_sugar_box"]
            )
            vis = input_func("visualize? (y/n)") if not vis_once else "y"
            while vis == "y":
                scene.fast_debug_vis(
                    interact=1,
                    collision_pt=vis_collision,
                    nonstop=vis_once,
                    write_video=write_video,
                )
                if vis_once:
                    break
                vis = input_func("visualize? (y/n)")
        else:
            print(
                "Set end_config example: -0.05929, -1.6124, -0.197, -2.53, -0.0952, 1.678, 0.587, 0.0398, 0.0398"
            )

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


if __name__ == "__main__":
    main()
