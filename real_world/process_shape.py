# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import os, math, sys
from os.path import *
import numpy as np
import numpy.random as npr

import os
import multiprocessing
import subprocess

try:
    from . import gen_xyz
    from . import gen_sdf
    from . import convert_sdf
    from . import gen_convex_shape
    from . import blender_process
except:
    pass
import IPython
import platform

PYTHON2 = True
if platform.python_version().startswith("3"):
    input_func = input
else:
    input_func = raw_input


def clean_file(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if (
                    file.endswith("model_normalized.obj")
                    or file.endswith("textured_simple.obj")
                    or file.endswith("png")
                    or file.endswith("npy")
                ):
                    continue

                file = os.path.join(dir, file)
                os.system("rm {}".format(file))
        else:
            print("not a dir", dir)


def rename_file(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if file.endswith("textured_simple.obj"):
                    file = os.path.join(dir, file)
                    new_file = file.replace("textured_simple", "model_normalized")
                    os.system("cp {} {}".format(file, new_file))
        else:
            print("not a dir", dir)


def cp_urdf(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if "model_normalized" in file:
                    file = os.path.join(dir, "model_normalized.urdf")
                    os.system("cp data/objects/model_normalized.urdf {}".format(file))
        else:
            print("not a dir", dir)


def cp_convex_urdf(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if "model_normalized" in file:
                    file = os.path.join(dir, "model_normalized.urdf")
                    os.system("cp data/objects/model_normalized.urdf {}".format(file))
        else:
            print("not a dir", dir)


import argparse

parser = argparse.ArgumentParser(
    "See README for used 3rd party packages to generate new mesh (importantly"
    "grasp is required to be used as target "
)
parser.add_argument(
    "-f",
    "--file",
    help="filename, needs to be in data/objects and contain model_normalized.obj",
    type=str,
    default="003_cracker_box",
)
parser.add_argument("-a", "--all", help="generate all", action="store_true")
parser.add_argument("-c", "--clean", help="clean all", action="store_true")
parser.add_argument("--xyz", help="generate extent and point file", action="store_true")
parser.add_argument("--sdf", help="generate sdf", action="store_true")
parser.add_argument(
    "--convex", help="convexify objects for bullet", action="store_true"
)
parser.add_argument(
    "--blender", help="use blender to fix topology for broken mesh", action="store_true"
)
parser.add_argument("--urdf", help="copy uniform urdf for bullet", action="store_true")
parser.add_argument(
    "--rename",
    help="rename to model_normalized file (shared name structure)",
    action="store_true",
)

args = parser.parse_args()

root_dir = "data/objects/"
if os.path.isdir(args.file):
    class_id = [
        os.path.join(args.file, obj)
        for obj in os.listdir(args.file)
        if os.path.isdir(os.path.join(args.file, obj))
    ]
else:
    class_id = [os.path.join(root_dir, args.file)]

if args.clean:
    print("clear all")
    clean_file(class_id)

print("files we have:", class_id)
generate = args.all or args.rename
if generate:
    print("rename file <======================")
    rename_file(class_id)

generate = args.all or args.xyz
if generate:
    gen_xyz.generate_extents_points(random_paths=class_id)


####### The object SDF is required for CHOMP Scene
generate = args.all or args.sdf
if generate:
    print("generate sdf (dont fail this one) <======================")
    gen_sdf.gen_sdf(random_paths=class_id)

generate = args.all or args.sdf
if generate:
    print("convert sdf <======================")
    convert_sdf.convert_sdf(class_id)


####### These two are mainly for rendering and simulation, needs update urdf if used in bullet
####### This can be used for meshes with broken topology and add textures uvs
try:
    generate = args.all or args.blender
    if generate:
        print("blender fix <======================")
        blender_process.process_obj(class_id)
except:
    print(
        "=================> need bpy or blender installed, (python 3.7 and see README)"
    )

####### The convex shape can be used for bullet.
try:
    generate = args.all or args.convex
    if generate:
        print("convexify object <======================")
        gen_convex_shape.convexify_model_subprocess(class_id)
except:
    print("=================> need vhacd from bullet, see README")

generate = args.all or args.urdf
if generate:
    print("copy uniform urdf <======================")
    cp_urdf(class_id)
