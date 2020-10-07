colormap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib

_colormap_cache = {}
import numpy as np


def _build_colormap(name, num_bins=256):
    base = cm.get_cmap(name)
    color_list = base(np.linspace(0, 1, num_bins))
    cmap_name = base.name + str(num_bins)
    colormap = LinearSegmentedColormap.from_list(cmap_name, color_list, num_bins)
    colormap = np.array(colormap(np.linspace(0, 1, num_bins)), dtype=np.float32)[:, :3]
    return colormap


def get_colormap(name):
    if name not in _colormap_cache:
        _colormap_cache[name] = _build_colormap(name)
    return _colormap_cache[name]


def colorize_tensor(tensor, cmap="magma", cmin=0, cmax=1):
    if len(tensor.shape) > 4:
        tensor = tensor.reshape(-1, *tensor.shape[-3:])
    if len(tensor.shape) == 2:
        tensor = tensor[None]
    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, (1))
    tensor = (tensor - cmin) / (cmax - cmin)
    tensor = np.clip((tensor * 255), 0.0, 255.0).astype(np.int)
    colormap = get_colormap(cmap)
    colorized = colormap[tensor].transpose(0, 3, 1, 2)
    return colorized


def colorize_depth(depth):
    if depth.min().item() < -0.1:
        return colorize_tensor(depth / 2.0 + 0.5)
    else:
        return colorize_tensor(depth, cmin=depth.min(), cmax=depth.max())


def colorize_numpy(array, to_byte=True):
    colorized = colorize_depth(array)
    colorized = np.squeeze(colorized).transpose(1, 2, 0)
    if to_byte:
        colorized = (colorized * 255).astype(np.uint8)
    return colorized


def get_color_mask(object_index, nc=None):
    """"""
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255

    return color_mask


def get_mask_colors(num):
    return list(get_color_mask(np.arange(num) + 1) / 255.0)
