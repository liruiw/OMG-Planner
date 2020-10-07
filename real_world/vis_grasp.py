# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from . import _init_paths
from omg.core import *
from omg import config
from ycb_render.ycb_renderer import YCBRenderer
import argparse

np.random.seed(233)

parser = argparse.ArgumentParser(description="clean up generating data")
parser.add_argument(
    "--target_name", help="target name", type=str, default="004_sugar_box"
)
parser.add_argument("--sample_num", help="target name", type=int, default=100)

args = parser.parse_args()
model_root_dir = config.cfg.root_dir + "data/objects"
grasp_root_dir = config.cfg.root_dir + "data/grasps/simulated/"

V = np.eye(4)
V[:3, 3] = 0, 0, 2

hand_anchor_points, line_index = get_hand_anchor_index_point()
renderer = YCBRenderer(width=640, height=480, gpu_id=0, render_marker=False)
nvidia_path = os.path.join(grasp_root_dir, "{}.npy".format(args.target_name))

try:
    nvidia_grasp = np.load(nvidia_path, allow_pickle=True)
    pose_grasp = nvidia_grasp.item()["transforms"]
except:
    nvidia_grasp = np.load(
        nvidia_path, allow_pickle=True, fix_imports=True, encoding="bytes"
    )
    pose_grasp = nvidia_grasp.item()[b"transforms"]
flip_pose = np.array(rotZ(np.pi / 2))
pose_grasp = np.matmul(pose_grasp, flip_pose)  # flip x, y

line_starts = []
line_ends = []

sample_idx = np.random.choice(np.arange(pose_grasp.shape[0]), args.sample_num).astype(
    np.int
)
hand_pose = pose_grasp[sample_idx]
hand_points = (
    np.matmul(hand_pose[:, :3, :3], hand_anchor_points.T) + hand_pose[:, :3, [3]]
)
hand_points = hand_points.transpose([1, 0, 2])
p1 = hand_points[:, :, line_index[0]].reshape([3, -1])
p2 = hand_points[:, :, line_index[1]].reshape([3, -1])

p1 = np.concatenate(([p1]), axis=1)
p2 = np.concatenate(([p2]), axis=1)

line_starts.append(p1)
line_ends.append(p2)
obj_path = [os.path.join(model_root_dir, args.target_name, "model_normalized.obj")]
renderer.load_objects(obj_path, [""], [(255, 0, 0)])
renderer.set_light_pos([0, 0, 1])
renderer.set_camera_default()
renderer.set_projection_matrix(640, 480, 525, 525, 319.5, 239.5, 0.1, 6)


img = renderer.vis(
    [pack_pose(np.eye(4))],
    [0],
    shifted_pose=np.eye(4),
    interact=2,
    visualize_context={
        "text": [""],
        "thickness": [1],
        "line": [(line_starts[0], line_ends[0])],
        "line_color": [[0, 255, 0]],
        "line_shader": True,
    },
)
renderer.release()
