# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse, time, sys

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import torch
import IPython
from subprocess import call


class SignedDensityField(object):
    """Data is stored in the following way
        data[x, y, z]
    update to integrate torch
    """

    def __init__(self, data, origin, delta):
        self.data = data
        self.nx, self.ny, self.nz = data.shape
        self.origin = origin
        self.delta = delta
        self.min_coords = origin
        self.max_coords = self.origin + delta * np.array(data.shape)

        self.data_torch = torch.from_numpy(self.data.astype(np.float32)).cuda()
        self.delta_torch = torch.FloatTensor([delta]).cuda().float()
        self.origin_torch = torch.from_numpy(self.origin).cuda().float()
        self.min = torch.LongTensor([0, 0, 0]).cuda()
        self.max = torch.LongTensor([self.nx - 1, self.ny - 1, self.nz - 1]).cuda()

    def resize(self, ratio):
        self.data *= ratio
        self.data_torch *= ratio

        self.delta *= ratio
        self.origin *= ratio

        self.delta_torch *= ratio
        self.origin_torch *= ratio

    def _rel_pos_to_idxes(self, rel_pos):
        i_min = np.array([0, 0, 0], dtype=np.int)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int)
        idx = ((rel_pos - self.origin) / self.delta).astype(int)
        return np.clip(idx, i_min, i_max)

    def _rel_pos_to_idxes_torch(self, rel_pos):
        return torch.min(
            torch.max(
                ((rel_pos - self.origin_torch) / self.delta_torch).long(), self.min
            ),
            self.max,
        )

    def get_distance(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        return self.data[idxes[..., 0], idxes[..., 1], idxes[..., 2]]

    def get_distance_torch(self, rel_pos):
        idxes = self._rel_pos_to_idxes_torch(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        return self.data_torch[idxes[:, 0], idxes[:, 1], idxes[:, 2]]

    def get_distance_grad(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        i_min = np.array([0, 0, 0], dtype=np.int)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int)
        neighbor1 = np.clip(idxes + 1, i_min, i_max)
        neighbor2 = np.clip(idxes - 1, i_min, i_max)
        dx = (
            self.data[neighbor1[..., 0], idxes[..., 1], idxes[..., 2]]
            - self.data[neighbor2[..., 0], idxes[..., 1], idxes[..., 2]]
        ) / (2 * self.delta)

        dy = (
            self.data[idxes[..., 0], neighbor1[..., 1], idxes[..., 2]]
            - self.data[idxes[..., 0], neighbor2[..., 1], idxes[..., 2]]
        ) / (2 * self.delta)

        dz = (
            self.data[idxes[..., 0], idxes[..., 1], neighbor1[..., 2]]
            - self.data[idxes[..., 0], idxes[..., 1], neighbor2[..., 2]]
        ) / (2 * self.delta)
        return np.stack([dx, dy, dz], axis=-1)

    def get_distance_grad_torch(self, rel_pos):
        idxes = self._rel_pos_to_idxes_torch(rel_pos)
        neighbor1 = torch.min(torch.max(idxes + 1, self.min), self.max)
        neighbor2 = torch.min(torch.max(idxes - 1, self.min), self.max)
        dx = (
            self.data_torch[neighbor1[..., 0], idxes[..., 1], idxes[..., 2]]
            - self.data_torch[neighbor2[..., 0], idxes[..., 1], idxes[..., 2]]
        ) / (2 * self.delta)

        dy = (
            self.data_torch[idxes[..., 0], neighbor1[..., 1], idxes[..., 2]]
            - self.data_torch[idxes[..., 0], neighbor2[..., 1], idxes[..., 2]]
        ) / (2 * self.delta)

        dz = (
            self.data_torch[idxes[..., 0], idxes[..., 1], neighbor1[..., 2]]
            - self.data_torch[idxes[..., 0], idxes[..., 1], neighbor2[..., 2]]
        ) / (2 * self.delta)
        return torch.stack([dx, dy, dz], dim=-1)

    def trim(self, dim, center=True):
        x_padding = (self.nx - dim) / 2
        y_padding = (self.ny - dim) / 2
        z_padding = (self.nz - dim) / 2
        if center:
            assert min(min(x_padding, y_padding), z_padding) >= 0
            self.data = self.data[
                x_padding : self.nx - x_padding,
                y_padding : self.ny - y_padding,
                z_padding : self.nz - z_padding,
            ]
        self.data = self.data[:dim, :dim, :dim]  # even odd
        self.origin = (
            self.origin + np.array([x_padding, y_padding, z_padding]) * self.delta
        )
        self.nx, self.ny, self.nz = dim, dim, dim
        self.max_coords = self.origin + self.delta * np.array(self.data.shape)
        self.data_torch = torch.from_numpy(self.data.astype(np.float32)).cuda()
        self.origin_torch = torch.from_numpy(self.origin).cuda().float()
        self.max = torch.LongTensor([self.nx - 1, self.ny - 1, self.nz - 1]).cuda()

    def dump(self, pkl_file):
        data = {}
        data["data"] = self.data
        data["origin"] = self.origin
        data["delta"] = self.delta
        pickle.dump(data, open(pkl_file, "wb"), protocol=2)

    def visualize(self, max_dist=0.1):
        try:
            from mayavi import mlab
        except:
            print("mayavi is not installed!")

        figure = mlab.figure("Signed Density Field")
        SCALE = 100  # The dimensions will be expressed in cm for better visualization.
        data = np.copy(self.data)
        data = np.minimum(max_dist, data)
        xmin, ymin, zmin = SCALE * self.origin
        xmax, ymax, zmax = SCALE * self.max_coords
        delta = SCALE * self.delta
        xi, yi, zi = np.mgrid[xmin:xmax:delta, ymin:ymax:delta, zmin:zmax:delta]
        data[data <= 0] -= 0.5  # 0.2

        data = -data
        xi = xi[: data.shape[0], : data.shape[1], : data.shape[2]]
        yi = yi[: data.shape[0], : data.shape[1], : data.shape[2]]
        zi = zi[: data.shape[0], : data.shape[1], : data.shape[2]]
        grid = mlab.pipeline.scalar_field(xi, yi, zi, data)
        vmin = np.min(data)
        vmax = np.max(data)
        mlab.pipeline.volume(grid, vmin=vmin, vmax=(vmax + vmin) / 2)
        mlab.axes()
        mlab.show()

    @classmethod
    def from_sdf(cls, sdf_file):

        with open(sdf_file, "r") as file:
            axis = 2
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
        return cls(data, np.array([x0, y0, z0]), delta)

    @classmethod
    def from_pth(cls, sdf_file):
        sdf = torch.load(sdf_file)
        max_coords = sdf["max_coords"].numpy()
        min_coords = sdf["min_coords"].numpy()
        data = sdf["sdf_torch"][0, 0].permute(1, 0, 2).numpy()
        delta = sdf["delta"]
        return cls(data, min_coords, delta)

    @classmethod
    def from_pkl(cls, pkl_file):
        data = pickle.load(open(pkl_file, "r"))
        return cls(data["data"], data["origin"], data["delta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Path to .sdf or .pkl file.",
        default="../robot_model/panda_arm/hand_finger.sdf",
        required=False,
    )
    parser.add_argument(
        "-v", action="store_true", help="Visualize signed density field."
    )
    parser.add_argument(
        "-n", action="store_true", help="Visualize signed density field as voxels."
    )
    parser.add_argument("--export", "-e", help="Export to a pickle file.")

    args = parser.parse_args()
    filename = args.file
    if filename.endswith(".sdf"):
        sdf = SignedDensityField.from_sdf(filename)
    elif filename.endswith(".pkl"):
        sdf = SignedDensityField.from_pkl(filename)
    elif filename.endswith(".pth"):
        sdf = SignedDensityField.from_pth(filename)

    print(
        "sdf info:",
        sdf.delta,
        sdf.data.shape,
        sdf.origin,
        (sdf.data > 0.01).sum(),
        sdf.delta * np.array(sdf.data.shape),
    )
    if args.v:
        sdf.visualize()
    if args.export:
        sdf.dump(args.export)
    if args.n:
        from voxel import voxel2obj

        voxel = (np.abs(sdf.data) <= 0.01).astype(np.int)
        voxel2obj("test.obj", voxel)
        call(["meshlab", "test.obj"])
