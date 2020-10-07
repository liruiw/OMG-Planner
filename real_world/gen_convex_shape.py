# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import sys
import os
import traceback
from multiprocessing import Process

USE_XVFP_SERVER = False
USE_SUBPROCESS = True
OVERWRITE = True


def convexify_model(obj):
    convex_obj = obj.replace(".obj", "_convex.obj")
    convex_stl = str(convex_obj).replace(".obj", ".stl")
    cmd = "test_vhacd_gmake_x64_release --input %s --output %s" % (
        obj,
        convex_obj,
    )  # for pybullet

    if not os.path.exists(obj):
        raise ValueError("No obj found : %s" % obj)

    # Stop printing command line output
    # with TemporaryFile() as f, stdout_redirected(f):
    os.system(cmd)
    cmd = "meshlabserver -i %s -o %s" % (convex_obj, convex_stl)
    os.system(cmd)


def convexify_model_subprocess(model_ids, cleanup=True):
    print(model_ids)

    for i, model_id in enumerate(model_ids):

        if cleanup:
            for f in os.listdir(model_id):

                if "convex" in f:
                    cmd = "rm " + os.path.join(model_id, f)
                    os.system(cmd)

        model_fn = os.path.join(model_id, "model_normalized.obj")
        print("convexifying %d/%d: %s" % (i + 1, len(model_ids), model_fn))

        sys.stdout.flush()  # To push print while running inside a Process
        convexify_model(model_fn)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def convex_decomp(data_path="data/objects/", random_paths=None):

    # Use binvox server
    if random_paths is not None:
        model_paths = random_paths
        pl = []
        numberOfThreads = 8
        obj_paths = []
        for idx, model_path in enumerate(model_paths):
            obj_paths.append(model_path)

        pl = []
        if USE_SUBPROCESS:
            for path in chunks(obj_paths, numberOfThreads):
                p = Process(
                    target=convexify_model_subprocess, args=[object_path, path, ""]
                )
                pl.append(p)
            for i in chunks(pl, numberOfThreads):
                for p in i:
                    p.start()
                for p in i:
                    p.join()
    else:
        model_paths = os.listdir(data_path)
        convexify_model_subprocess(object_path, data_path)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        convexify_model(sys.argv[1])
    elif len(sys.argv) > 1:
        convexify_model(sys.argv[1])
    else:
        convex_decomp()
