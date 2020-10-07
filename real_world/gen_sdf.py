# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import multiprocessing
import subprocess
import IPython
import numpy as np
PADDING = 20


def generate_sdf(path_to_sdfgen, obj_filename, delta, padding, dim):
    """ Converts mesh to an sdf object """

    # create the SDF using binary tools, avoid overwrite
    dummy_cmd = "cp %s %s" % (obj_filename, obj_filename.replace(".obj", ".dummy.obj"))
    os.system(dummy_cmd)
    sdfgen_cmd = '%s "%s" %f %d' % (
        path_to_sdfgen,
        obj_filename.replace(".obj", ".dummy.obj"),
        delta,
        padding,
    )
    os.system(sdfgen_cmd)
    sdf_filename = obj_filename.replace(".obj", ".dummy.sdf")
    sdf_dim_filename = obj_filename.replace(".obj", "_chomp.sdf")
    print("SDF Command: %s" % sdfgen_cmd)
    rename_cmd = "mv %s %s" % (sdf_filename, sdf_dim_filename)
    os.system(rename_cmd)
    clean_cmd = "rm %s; rm %s" % (
        obj_filename.replace(".obj", ".dummy.obj"),
        sdf_filename.replace(".sdf", ".vti"),
    )
    os.system(clean_cmd)
    print("Rename Output Location", sdf_dim_filename)
    return


def do_job_convert_obj_to_sdf(input):
    x, dim, extra_padding = input

    file = os.path.join(prefix, str(file_list_all[x]), surfix)
    extent_file = file.replace(".obj", ".extent.txt")
    sdf_file = file.replace(".obj", ".sdf")
    extent = np.loadtxt(extent_file).astype(np.float32)
    REG_SIZE = 0.2
    scale = np.max(extent) / REG_SIZE
    extra_padding = min(int(extra_padding * scale), 30)
    dim = max(min(dim * scale, 100), 32)
    delta = np.max(extent) / dim
    padding = extra_padding  #

    print("SDF delta: {} padding: {}".format(delta, padding))
    dim = dim + extra_padding * 2
    generate_sdf(path_sdfgen, file, delta, padding, dim)
    print("Done job number", x)


path_sdfgen = "sdf_gen"
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_list_all = [sys.argv[1]]
        prefix = ""
        surfix = ""
        dim = 32
        pad = 8
        if len(sys.argv) > 2:
            pad = int(sys.argv[2])

        do_job_convert_obj_to_sdf((0, dim, pad))


def gen_sdf(random_paths=None, dim=32):  # modify dim
    global prefix, surfix, file_list_all
    surfix = "model_normalized.obj"

    file_list_all = random_paths
    prefix = ""

    object_numbers = len(file_list_all)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores // 3)
    padding = 20  # one with padding, one without
    param_list = [
        list(a)
        for a in zip(
            range(object_numbers), [dim] * object_numbers, [padding] * object_numbers
        )
    ]
    pool.map(do_job_convert_obj_to_sdf, param_list)  # 32, 40
