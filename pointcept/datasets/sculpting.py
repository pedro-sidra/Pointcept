import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping
from pointcept.datasets.sculpting_ops import (
    add_random_cubes,
    get_random_cubes_random_sampled_point_references,
)


from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class VoxelizeAgg(object):
    pass


@TRANSFORMS.register_module()
class SculptingOcclude(object):
    def __init__(
        self,
        cube_size_min=0.1,
        cube_size_max=0.5,
        npoint_frac=0.005,
        npoints=None,
        cell_size=0.02,
        density_factor=0.1,
        kill_color_proba=0.5,
        sampling="random",
    ):
        self.cube_size_min = cube_size_min
        self.cube_size_max = cube_size_max
        self.npoint_frac = npoint_frac
        self.npoints = npoints
        self.cell_size = cell_size
        self.density_factor = density_factor
        self.kill_color_proba = kill_color_proba
        self.sampling = sampling

    def get_random_colors(self, size, low=0, high=255):
        return np.random.randint(low, high, size).astype(np.float32)

    def get_random_normals(self, size):
        n = np.random.rand(*size).astype(np.float32) * 2 - 1
        n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
        return n

    def add_random_cubes(self, data_dict):

        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        semantic_label = data_dict.get("segment", np.ones(len(xyz), dtype=int))
        instance_label = data_dict.get("instance", -1 * np.ones(len(xyz), dtype=int))

        if self.npoints is None:
            ncubes = int(self.npoint_frac * len(xyz))
        else:
            ncubes = self.npoints

        cubes = get_random_cubes_random_sampled_point_references(
            self.cube_size_min,
            self.cube_size_max,
            xyz,
            npoints=ncubes,
            cell_size=self.cell_size,
            actual_cube=False,
            sphere=False,
            point_sampling=self.sampling,
            density_factor=self.density_factor,
        )

        xyz = np.vstack([xyz, cubes])

        rand_colors = self.get_random_colors(cubes.shape)
        rgb = np.vstack([rgb, rand_colors])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, rand_normals])

        # Randomly turn colors off
        # if np.random.rand() < self.kill_color_proba:
        #     rgb = rgb * 0.0

        semantic_label = np.hstack(
            [
                np.ones_like(semantic_label, dtype=np.int32),
                np.zeros(cubes.shape[0], dtype=np.int32),
            ]
        )

        instance_label = np.hstack(
            [
                -1 * np.ones(instance_label.shape[0], dtype=np.int32),
                -1 * np.ones(cubes.shape[0], dtype=np.int32),
            ]
        )

        return xyz, rgb, semantic_label, instance_label, normal

    def __call__(self, data_dict):
        """
        for semseg models,
        data_dict.keys() = ['coord', 'color', 'normal', 'name', 'segment', 'instance']
        """

        (
            data_dict["coord"],
            data_dict["color"],
            data_dict["segment"],
            data_dict["instance"],
            data_dict["normal"],
        ) = self.add_random_cubes(data_dict)
        # from pointcept.utils import ForkedPdb; ForkedPdb().set_trace()
        return data_dict
