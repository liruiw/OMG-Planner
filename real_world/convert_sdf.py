# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np

import time
import argparse
import os


def read_sdf(sdf_file):
    with open(sdf_file, "r") as file:
        lines = file.readlines()
        nx, ny, nz = map(int, lines[0].split(" "))
        x0, y0, z0 = map(float, lines[1].split(" "))
        delta = float(lines[2].strip())
        data = np.zeros([nx, ny, nz])
        for i, line in enumerate(lines[3:]):
            idx = i % nx
            idy = int(i / nx) % ny
            idz = int(i / (nx * ny))
            val = float(line.strip())
            data[idx, idy, idz] = val
    return (data, np.array([x0, y0, z0]), delta)


def convert_sdf(sdf_files):

    time_start = time.time()

    for sdf_file in sdf_files:
        print(" start converting sdf for {} ... ".format(sdf_file))

        sdf_file = os.path.join(sdf_file, "model_normalized_chomp.sdf")
        sdf_info = read_sdf(sdf_file)
        sdf = sdf_info[0]
        min_coords = sdf_info[1]
        delta = sdf_info[2]
        max_coords = min_coords + delta * np.array(sdf.shape)
        xmin, ymin, zmin = min_coords
        xmax, ymax, zmax = max_coords
        sdf_torch = (
            torch.from_numpy(sdf).float().permute(1, 0, 2).unsqueeze(0).unsqueeze(1)
        )
        time_elapse = time.time() - time_start

        print(
            "     sdf size = {}x{}x{}".format(
                sdf_torch.size(2), sdf_torch.size(3), sdf_torch.size(4)
            )
        )
        print(
            "     minimal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm".format(
                xmin * 100, ymin * 100, zmin * 100
            )
        )
        print(
            "     maximal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm".format(
                xmax * 100, ymax * 100, zmax * 100
            )
        )
        print(" finished loading sdf, time elapsed = {} ! ".format(time_elapse))

        save_path = sdf_file[:-4] + ".pth"
        print(" saving sdf to {}! ".format(save_path))
        torch.save(
            {
                "min_coords": torch.from_numpy(min_coords),
                "max_coords": torch.from_numpy(max_coords),
                "delta": delta,
                "sdf_torch": sdf_torch,
            },
            save_path,
        )
        print(" finished saving sdf ! ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--file", "-f", help="Path to .sdf file.", required=True)
    args = parser.parse_args()
    sdf_file = args.file
    convert_sdf([sdf_file])
