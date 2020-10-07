# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cv2
import numpy as np
from pyassimp import *
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from multiprocessing import Process
import time
import IPython


def homotrans(M, p):
    p = np.asarray(p)
    if p.shape[-1] == M.shape[1] - 1:
        p = np.append(p, np.ones_like(p[..., :1]), -1)
    p = np.dot(p, M.T)
    return p[..., :-1] / p[..., -1:]


def recursive_load(node, vertices, normals, scale):
    if node.meshes:
        transform = node.transformation
        for idx, mesh in enumerate(node.meshes):
            if mesh.faces.shape[-1] != 3:
                continue
            mesh_vertex = homotrans(transform, mesh.vertices)
            vertices.append(mesh_vertex)

            # compute face normal for simplicity
            if mesh.normals.shape[0] > 0:
                mesh_normals = (
                    transform[:3, :3].dot(mesh.normals.transpose()).transpose()
                )
            else:
                mesh_normals = np.zeros_like(mesh_vertex)
                mesh_normals[:, -1] = 1
            normals.append(mesh_normals)
    for child in node.children:
        recursive_load(child, vertices, normals, scale)
    return vertices, normals


def gen_normal_xyz(path, pt_num=3000):
    print(path)
    extent_filename = path.replace(".obj", ".extent.txt")
    offset_filename = path.replace(".obj", ".offset.txt")

    scene = load(path)
    vertices, normals = recursive_load(scene.rootnode, [], [], 1)
    vertices = np.concatenate(vertices, axis=0)

    normals = np.concatenate(normals, axis=0)
    normals /= np.linalg.norm(normals + 1e-12, axis=1, keepdims=True)
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    center = np.mean(vertices, axis=0)  # center at 0

    extent_f = open(extent_filename, "w")  # continue writing
    offset_f = open(offset_filename, "w")
    extent_f.write("%f %f %f\n" % (extent[0], extent[1], extent[2]))
    offset_f.write("%f %f %f\n" % (center[0], center[1], center[2]))
    idx = np.random.choice(
        np.arange(vertices.shape[0]), min(pt_num, vertices.shape[0]), replace=False
    )
    xyz = np.concatenate(
        (vertices[idx], normals[idx]), axis=-1
    )  # subtract the offset - center

    xyz_file = path.replace(".obj", ".xyz")
    np.savetxt(xyz_file, xyz, fmt="%f")


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def generate_extents_points(random_paths=None):
    USE_SUBPROCESS = True
    np.random.seed(1)
    model_paths = random_paths
    numberOfThreads = 32
    obj_paths = []
    for idx, model_path in enumerate(model_paths):
        obj_paths.append(os.path.join(model_path, "model_normalized.obj"))

    pl = []

    if USE_SUBPROCESS:
        for i, path in tqdm(enumerate(obj_paths)):
            p = Process(target=gen_normal_xyz, args=[path])
            pl.append(p)
        for i in chunks(pl, numberOfThreads):
            for p in i:
                p.start()
            for p in i:
                p.join()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        gen_normal_xyz(sys.argv[1])
