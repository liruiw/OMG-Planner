import numpy as np
import os


def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, 1, :, :]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, 1, :, :]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, 1, :, :]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    num_fn = np.sum(
        np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :])
    )  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])


def voxel2mesh(voxels, surface_view, scale=0.01):
    cube_verts = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]  # 8 points

    cube_faces = [
        [0, 1, 2],
        [1, 3, 2],
        [2, 3, 6],
        [3, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [0, 5, 1],
        [0, 4, 5],
        [6, 7, 5],
        [6, 5, 4],
        [1, 7, 3],
        [1, 5, 7],
    ]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(np.abs(voxels) > 0.1)  # np.where(voxels > 0.3)
    voxels[positions] = 1
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face
        if (
            not surface_view
            or np.sum(voxels[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]) < 27
        ):
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, "w") as f:
        # write vertices
        f.write("g\n# %d vertex\n" % len(verts))
        for vert in verts:
            f.write("v %f %f %f\n" % tuple(vert))

        # write faces
        f.write("# %d faces\n" % len(faces))
        for face in faces:
            f.write("f %d %d %d\n" % tuple(face))


def voxel2obj(filename, pred, surface_view=True, write=True, scale=0.01):
    verts, faces = voxel2mesh(pred, surface_view, scale)
    if write:
        write_obj(filename, verts, faces)
    return verts, faces


def voxelize_model_binvox(obj, n_vox, return_voxel=True, binvox_add_param=""):

    cmd = "./binvox -d %d -cb -dc -aw -pb %s -t binvox %s" % (
        n_vox,
        binvox_add_param,
        obj,
    )

    if not os.path.exists(obj):
        raise ValueError("No obj found : %s" % obj)

    # Stop printing command line output
    with TemporaryFile() as f, stdout_redirected(f):
        os.system(cmd)

    # load voxelized model
    if return_voxel:
        with open("%s.binvox" % obj[:-4], "rb") as f:
            vox = binvox_rw.read_as_3d_array(f)

        return vox.data, vox.scale / vox.dims[0]
