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
    get_random_colored_cubes_on_pts,
)
from copy import deepcopy


from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class VoxelizeAgg(object):
    def __init__(
        self,
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        mode_agg_vars=[],
        mean_agg_vars=[],
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

        self.mode_agg_vars = mode_agg_vars
        self.mean_agg_vars = mean_agg_vars

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()

        # To voxel indexes
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord

        # Save the min coord in original values
        min_coord = min_coord * np.array(self.grid_size)

        # Hash of the grid coords -> to group the unique voxel coords
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        # unique values of the key
        # inverse: mapping from points to voxels (p2v_map)
        # count: points per voxel
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # mapping from voxels to a single point (v2p_map)
        first_point_idx = idx_sort[np.cumsum(np.insert(count, 0, 0)[0:-1])]

        for var_name in (*self.mode_agg_vars, *self.mean_agg_vars):
            var = data_dict[var_name]
            var_voxel = np.empty(shape=(len(count), *var.shape[1:]), dtype=var.dtype)
            for i in np.unique(count):
                (voxel_ids,) = np.where(count == i)
                point_locs = np.isin(inverse, voxel_ids)

                var_by_voxel = var[idx_sort][point_locs].reshape(-1, *var.shape[1:], i)

                if var_name in self.mode_agg_vars:
                    # TODO: move back to mode
                    # if i == 1 or i == 2:
                    #     var_voxel[voxel_ids] = var_by_voxel[:, 0]
                    # else:
                    var_voxel[voxel_ids] = np.max(var_by_voxel, axis=-1)
                elif var_name in self.mean_agg_vars:
                    var_voxel[voxel_ids] = np.mean(var_by_voxel, axis=-1)
            data_dict[var_name] = var_voxel

        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[first_point_idx]
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        # for key in self.keys:
        #     data_dict[key] = data_dict[key][idx_unique]
        return data_dict

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    @classmethod
    def array_mode(self, ndarray, axis=0, return_details=False):
        # Check inputs
        ndarray = np.asarray(ndarray)
        ndim = ndarray.ndim
        if ndarray.size == 1:
            return (ndarray[0], 1)
        elif ndarray.size == 0:
            raise Exception("Cannot compute mode on empty array")
        try:
            axis = range(ndarray.ndim)[axis]
        except:
            raise Exception(
                'Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim)
            )

        # If array is 1-D and np version is > 1.9 np.unique will suffice
        if all(
            [
                ndim == 1,
                int(np.__version__.split(".")[0]) >= 1,
                int(np.__version__.split(".")[1]) >= 9,
            ]
        ):
            modals, counts = np.unique(ndarray, return_counts=True)
            index = np.argmax(counts)
            return modals[index], counts[index]

        # Sort array
        sort_idx = np.argsort(ndarray, axis=axis)
        sort = np.take_along_axis(ndarray, sort_idx, axis=axis)
        inverse_sort_idx = np.take_along_axis(
            np.indices(ndarray.shape)[axis], sort_idx, axis=axis
        )
        # Create array to transpose along the axis and get padding shape
        transpose = np.roll(np.arange(ndim)[::-1], axis)
        shape = list(sort.shape)
        shape[axis] = 1
        # Create a boolean array along strides of unique values
        strides = (
            np.concatenate(
                [
                    np.zeros(shape=shape, dtype="bool"),
                    np.diff(sort, axis=axis) == 0,
                    np.zeros(shape=shape, dtype="bool"),
                ],
                axis=axis,
            )
            .transpose(transpose)
            .ravel()
        )
        # Count the stride lengths
        counts = np.cumsum(strides)
        counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
        counts[strides] = 0
        # Get shape of padded counts and slice to return to the original shape
        shape = np.array(sort.shape)
        shape[axis] += 1
        shape = shape[transpose]
        slices = [slice(None)] * ndim
        slices[axis] = slice(1, None)
        # Reshape and compute final counts
        counts = counts.reshape(shape).transpose(transpose)[tuple(slices)] + 1

        # Find maximum counts and return modals/counts
        slices = [slice(None, i) for i in sort.shape]
        del slices[axis]
        index = np.ogrid[slices]
        index.insert(axis, np.argmax(counts, axis=axis))

        reverse_index = deepcopy(index)
        reverse_index[axis] = inverse_sort_idx[tuple(index)]

        index = tuple(index)
        reverse_index = tuple(reverse_index)
        if return_details:
            return ndarray[reverse_index], counts[index], reverse_index
        else:
            return sort[index]


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

        if self.npoints is None:
            ncubes = int(self.npoint_frac * len(xyz))
        else:
            ncubes = self.npoints

        cubes, cube_feats = get_random_colored_cubes_on_pts(
            self.cube_size_min,
            self.cube_size_max,
            xyz,
            feats=rgb,
            npoints=ncubes,
            cell_size=self.cell_size,
            actual_cube=False,
            sphere=False,
            point_sampling=self.sampling,
            density_factor=self.density_factor,
        )

        xyz = np.vstack([xyz, cubes])

        # rand_colors = self.get_random_colors(cubes.shape)
        rgb = np.vstack([rgb, cube_feats])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, rand_normals])

        # Randomly turn colors off
        if np.random.rand() < self.kill_color_proba:
            rgb = rgb * 0.0

        semantic_label = np.hstack(
            [
                np.ones_like(semantic_label, dtype=np.int32),
                np.zeros(cubes.shape[0], dtype=np.int32),
            ]
        )

        return xyz, rgb, semantic_label, normal

    def __call__(self, data_dict):
        """
        for semseg models,
        data_dict.keys() = ['coord', 'color', 'normal', 'name', 'segment', 'instance']
        """

        (
            data_dict["coord"],
            data_dict["color"],
            data_dict["segment"],
            # data_dict["instance"],
            data_dict["normal"],
        ) = self.add_random_cubes(data_dict)
        # from pointcept.utils import ForkedPdb; ForkedPdb().set_trace()
        return data_dict
