import numpy as np
from typing import Tuple
from itertools import combinations
import scipy.spatial as spatial
import tqdm
from .c_allocate import (
    norm, argwhere_nonzeros,
    argwhere_value_double,
    any_value_double,
    duplicate_map,
    _allocate_patch,
    _allocate_angle,
)


_ANGLE_SCALE = 180.0 / np.pi


def _invoke_progress(progress_callback, progress=None, label=None, indeterminate=None):
    if progress_callback is None:
        return
    if progress is not None:
        progress = float(np.clip(progress, 0.0, 1.0))
    progress_callback(progress, label, indeterminate)


def _as_index_array(indices: np.ndarray) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _deduplicate_indices(indices: np.ndarray, unique_inverse: np.ndarray) -> np.ndarray:
    indices = _as_index_array(indices)
    if indices.shape[0] <= 1:
        return indices
    _, unique_positions = np.unique(unique_inverse[indices], return_index=True)
    if unique_positions.shape[0] == indices.shape[0]:
        return indices
    return indices[np.sort(unique_positions)]


# [TODO] the allocator needs to be accelerated to handle large point clouds
def allocate(*args: Tuple[np.ndarray, ...], min_patch_size: int = 10, max_patch_size: int = 20,
             overlap: float = 0.2, feature_angle: float = 30, progress_callback=None) -> list:
    """
    Create a list of patches from a set of points and normals, if provided, from a point cloud.

    Parameters
    ----------
        points : np.ndarray
            A numpy array of shape (n, d) containing the points of the point cloud.
        normals : np.ndarray
            A numpy array of shape (n, d) containing the normals of the point cloud.
        min_patch_size : int
            The minimum number of points in a patch.
        max_patch_size : int
            The maximum number of points in a patch.
        overlap : float
            The maximum percentage of overlap between patches as a value between 0 and 1.
        feature_angle : float
            The maximum angle allowed between point-wise normal vectors
    Returns
    -------
        patches : list
            A list of tuples containing the points and normals of each patch.
    """
    last_progress = 0.0

    def report(progress=None, label=None, indeterminate=None, force=False):
        nonlocal last_progress
        if progress is not None:
            progress = max(last_progress, float(np.clip(progress, 0.0, 1.0)))
            if not force and progress < 1.0 and (progress - last_progress) < 0.005:
                return
            last_progress = progress
        _invoke_progress(progress_callback, progress, label, indeterminate)

    if len(args) == 0:
        print("Error: No data provided.")
        return [None, None]
    elif len(args) == 1:
        points = args[0]
        normals = None
    else:
        points = args[0]
        normals = args[1]
        magnitudes = norm(normals)
        if any_value_double(magnitudes.flatten(), 0.0):
            e = argwhere_value_double(magnitudes.flatten(), 0.0)
            print("Error: Normals with zero magnitude found at indices:\n{}.".format(e))
            print("Deleting points with zero magnitude normals...")
            print("Points:\n{}".format(points[e, :]))
            print("Normals:\n{}".format(normals[e, :]))
            indices = argwhere_nonzeros(magnitudes.flatten())
            points = points[indices, :]
            normals = normals[indices, :]
            magnitudes = norm(normals)
        normals = normals / magnitudes

    report(0.0, "Preparing point cloud for patch allocation...", force=True)

    _, unique_inverse, unique_counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    duplicates, duplicate_set = duplicate_map(unique_inverse, unique_counts)
    has_duplicates = len(duplicates) > 0

    if normals is None:
        if has_duplicates:
            print("Warning: Duplicate points found with no normals provided.")
            print("Removing duplicate points...")
            print("Done.")
    elif has_duplicates:
        remove_duplicates = []
        for key in duplicates.keys():
            repeated_indices = duplicates[key]
            repeated_normals = normals[repeated_indices, :]
            comb = combinations(list(range(repeated_normals.shape[0])), 2)
            for i, j in comb:
                dot = np.dot(repeated_normals[i, :], repeated_normals[j, :])
                dot = np.clip(dot, -1, 1)
                angle = np.arccos(dot) * _ANGLE_SCALE
                if not np.isclose(angle, 0):
                    if angle < feature_angle:
                        feature_angle = angle
                else:
                    print("Error: Duplicate points with identical normals found.")
                    print('Points:\n{} \nRepeated Normals:\n{}\nAngle:{}\nCombination:{}'.format(
                        points[repeated_indices, :], repeated_normals, angle, tuple([i, j])))
                    print("Removing duplicate point and normal {}...".format(repeated_indices[j]))
                    remove_duplicates.append(repeated_indices[j])
                    duplicate_set.remove(repeated_indices[j])
                    duplicates[key].remove(repeated_indices[j])
        if len(remove_duplicates) > 0:
            keep_mask = np.ones(points.shape[0], dtype=bool)
            keep_mask[np.unique(np.asarray(remove_duplicates, dtype=np.int64))] = False
            points = points[keep_mask, :]
            normals = normals[keep_mask, :]

    _, unique_inverse, unique_counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    duplicates, duplicate_set = duplicate_map(unique_inverse, unique_counts)
    kdtree = spatial.cKDTree(points)
    overlap = np.clip(overlap, 0, 1)
    max_patch_size = np.min([max_patch_size, points.shape[0]])
    min_patch_size = np.min([min_patch_size, max_patch_size // 2])
    patch_points = []
    patch_normals = []
    point_set = set(np.arange(points.shape[0]).tolist())
    remaining_points = []
    _, neighbor_indices = kdtree.query(points, k=max_patch_size)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, np.newaxis]

    progress_bar = tqdm.tqdm(
        total=len(point_set),
        desc='Allocating patches',
        unit='point',
        leave=False,
        disable=progress_callback is not None,
    )
    processed_main = 0
    consumed_total = 0
    while len(point_set) > 0:
        progress_start = len(point_set)
        point_idx = point_set.pop()
        indices = neighbor_indices[point_idx, :]
        if normals is not None:
            indices = np.asarray(
                _allocate_angle(point_idx, indices, points[indices, :], normals[indices, :], feature_angle),
                dtype=np.int64,
            )
            indices = _deduplicate_indices(indices, unique_inverse)
        if len(indices) < min_patch_size:
            remaining_points.append(point_idx)
        else:
            patch_points.append(points[indices, :])
            if normals is not None:
                patch_normals.append(normals[indices, :])
            else:
                patch_normals.append(None)
            point_set = _allocate_patch(indices, overlap, point_set, duplicate_set)
        progress_end = len(point_set)
        consumed_this_seed = progress_start - progress_end
        processed_main += 1
        consumed_total += consumed_this_seed
        progress_bar.update(consumed_this_seed)

        avg_consumed_per_seed = consumed_total / processed_main
        estimated_main_remaining = len(point_set) / max(avg_consumed_per_seed, 1.0)
        deferred_rate = len(remaining_points) / processed_main
        estimated_remaining_work = len(remaining_points) + deferred_rate * estimated_main_remaining
        estimated_total_work = processed_main + estimated_main_remaining + estimated_remaining_work
        if estimated_total_work > 0:
            report(
                processed_main / estimated_total_work,
                "Allocating patches (estimating remaining seeds)...",
            )
    progress_bar.close()

    progress_bar = tqdm.tqdm(
        total=len(remaining_points),
        desc='Allocating remaining patches',
        unit='point',
        leave=False,
        disable=progress_callback is not None,
    )
    processed_remaining = 0
    total_work = processed_main + len(remaining_points)
    if total_work > 0 and len(remaining_points) > 0:
        report(processed_main / total_work, "Allocating small remaining patches...", force=True)
    while len(remaining_points) > 0:
        progress_start = len(remaining_points)
        point_idx = remaining_points.pop()
        indices = neighbor_indices[point_idx, :]
        if normals is not None:
            dots = np.dot(normals[point_idx, :], normals[indices, :].T)
            dots = np.clip(dots, -1, 1)
            angles = np.arccos(dots) * _ANGLE_SCALE
            if np.count_nonzero(angles < 180) >= 3:
                indices = np.asarray(
                    _allocate_angle(point_idx, indices, points[indices, :], normals[indices, :], feature_angle),
                    dtype=np.int64,
                )
                indices = _deduplicate_indices(indices, unique_inverse)
            else:
                indices = _deduplicate_indices(indices, unique_inverse)
                indices = indices[:3]
        patch_points.append(points[indices, :])
        if normals is not None:
            patch_normals.append(normals[indices, :])
        else:
            patch_normals.append(None)
        progress_end = len(remaining_points)
        progress_bar.update(progress_start - progress_end)
        processed_remaining += 1
        if total_work > 0:
            report(
                (processed_main + processed_remaining) / total_work,
                "Allocating small remaining patches...",
            )
    progress_bar.close()
    report(1.0, "Patch allocation complete", force=True)
    return list(zip(patch_points, patch_normals))
