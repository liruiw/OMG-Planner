# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys

import cv2
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython
import random
import time
from .panda_scene import PandaYCBEnv

np.set_printoptions(2)
plt.rcParams["figure.figsize"] = (15, 5)


def read_trajectory(scene_dir, scene_file):
    env = PandaYCBEnv(renders=True, egl_render=False)
    full_file = os.path.join(scene_dir, scene_file + ".mat")
    scene_mat = sio.loadmat(full_file)
    plan = scene_mat["trajectory"]
    init_joints = scene_mat["states"][0]
    env.reset()
    env.cache_reset(scene_file=full_file, init_joints=init_joints.tolist())

    for k in range(plan.shape[0]):
        obs, rew, done, _ = env.step(plan[k].tolist())

    rew = env.retract()
    print("success:", rew)
    env.disconnect()


def read_image(scene_dir, scene_file):
    plt.ion()
    plt.gca()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    img1 = np.zeros([640, 480, 3])
    img2 = np.zeros([640, 480])
    img3 = np.zeros([640, 480], dtype=np.int32)
    ax1.imshow(img1, interpolation="none", animated=True, label="info")
    ax2.imshow(img2, interpolation="none", animated=True, label="info")
    ax3.imshow(img3, interpolation="none", animated=True, label="info")

    file = os.path.join(scene_dir, scene_file)
    i = 0
    while os.path.exists(file + "_color_{}.png".format(i)):
        color = cv2.imread(file + "_color_{}.png".format(i))
        depth = (
            cv2.imread(file + "_depth_{}.png".format(i), cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            / 5000.0
        )
        mask = cv2.imread(file + "_mask_{}.png".format(i), cv2.IMREAD_UNCHANGED).astype(
            np.int32
        )

        ax1.imshow(color[..., ::-1])
        ax1.set_title("color max:{} min:{}".format(color.max(), color.min()))
        ax2.imshow(depth)
        ax2.set_title("depth max:{:.2f} min:{:.2f}".format(depth.max(), depth.min()))
        ax3.imshow(mask)
        ax3.set_title("mask max:{} min:{}".format(mask.max(), mask.min()))
        plt.show()
        plt.pause(0.00001)
        i += 1


def write_video(scene_dir, scene_file, output_dir="output_videos"):
    video_file = os.path.join(output_dir, scene_file + "_demonstration.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG
    video_writer = cv2.VideoWriter(video_file, fourcc, 10.0, (640, 480))
    file = os.path.join(scene_dir, scene_file)
    i = 0
    print(file, video_file, os.path.exists(file + "_color_{}.png".format(0)))
    while os.path.exists(file + "_color_{}.png".format(i)):
        color = cv2.imread(file + "_color_{}.png".format(i))
        video_writer.write(color)
        i += 1
    video_writer.release()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", help="traj, img", type=str, default="traj")
    parser.add_argument("-f", "--file", help="filename", type=str, default="scene_0")
    parser.add_argument(
        "-d", "--dir", help="directory", type=str, default="data/demonstrations"
    )
    args = parser.parse_args()
    file = args.file

    if args.option == "traj":
        read_trajectory(args.dir, file)
    if args.option == "img":
        read_image(args.dir, file)
    if args.option == "video":
        write_video(args.dir, file)
 