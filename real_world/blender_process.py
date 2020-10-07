# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import sys
from pathlib import Path
import bpy
import os

MAX_SIZE = 5e7


_package_dir = os.path.dirname(os.path.realpath(__file__))


def process_obj(file_path, strip_materials=False):
    if "kitchen" not in file_path:
        return

    path = os.path.join(file_path, "model_normalized.obj")
    processed_path = os.path.join(file_path, "model_normalized.processed.obj")

    if os.path.exists(processed_path) or not os.path.isdir(file_path):
        return
    model_size = os.path.getsize(path)
    # if model_size > MAX_SIZE:
    #     print("Model too big ({} > {})".format(model_size, MAX_SIZE))
    #     continue

    print(f"Processing {path!s}")
    # continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(
        filepath=str(path),
        use_edges=True,
        use_smooth_groups=True,
        use_split_objects=True,
        use_split_groups=True,
        use_groups_as_vgroups=False,
        use_image_search=True,
    )

    if len(bpy.data.objects) > 10:
        print("Too many objects. Skipping for now..")
        return

    if strip_materials:
        print("Deleting materials.")
        for material in bpy.data.materials:
            material.user_clear()
            bpy.data.materials.remove(material)

    for obj_idx, obj in enumerate(bpy.data.objects):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        print("Clearing split normals and removing doubles.")
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.normals_make_consistent(inside=False)

        print("Unchecking auto_smooth")
        obj.data.use_auto_smooth = False

        bpy.ops.object.modifier_add(type="EDGE_SPLIT")
        print("Adding edge split modifier.")
        mod = obj.modifiers["EdgeSplit"]
        mod.split_angle = 20

        bpy.ops.object.mode_set(mode="OBJECT")

        print("Applying smooth shading.")
        bpy.ops.object.shade_smooth()

        print("Running smart UV project.")
        bpy.ops.uv.smart_project()

        bpy.context.active_object.select_set(state=False)

    bpy.ops.export_scene.obj(
        filepath=str(processed_path),
        group_by_material=True,
        keep_vertex_order=True,
        use_normals=True,
        use_uvs=True,
        use_materials=True,
        check_existing=False,
    )
    print("Saved to {}".format(processed_path))


def main():
    # Drop blender arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument("--strip-materials", action="store_true")
    parser.add_argument(
        "-f", "--file", help="filename", type=str, default="003_cracker_box"
    )
    args = parser.parse_args()
    root_dir = "../data/objects/"
    if os.path.isdir(args.file):
        class_id = [os.path.join(args.file, obj) for obj in os.listdir(args.file)]
    else:
        class_id = [os.path.join(root_dir, args.file)]

    for i, file_path in enumerate(class_id):
        process_obj(file_path, strip_materials=args.strip_materials)


if __name__ == "__main__":
    main()
