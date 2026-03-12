import math
import numpy as np
from typing import Dict, List, Set


_RAD_TO_DEG = 180.0 / math.pi

def norm(data: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data, axis=1, keepdims=True)


def argwhere_nonzeros(data: np.ndarray) -> np.ndarray:
    return np.nonzero(data)[0].astype(np.int64)


def argwhere_value_double(data: np.ndarray, value: float) -> np.ndarray:
    return np.where(data == value)[0].astype(np.int64)


def argwhere_value_int(data: np.ndarray, value: int) -> np.ndarray:
    return np.where(data == value)[0].astype(np.int64)


def any_value_double(data: np.ndarray, value: float) -> bool:
    return bool(np.any(data == value))


def any_value_int(data: np.ndarray, value: int) -> bool:
    return bool(np.any(data == value))


def c_dict(data: np.ndarray) -> Dict[int, List[float]]:
    return {i: list(map(float, row)) for i, row in enumerate(data)}


def duplicate_map(unique_inverse: np.ndarray, unique_counts: np.ndarray):
    dup: Dict[int, List[int]] = {}
    dup_set: Set[int] = set()
    for idx, key in enumerate(unique_inverse):
        int_key = int(key)
        if unique_counts[int_key] <= 1:
            continue
        dup.setdefault(int_key, []).append(idx)
        dup_set.add(idx)
    return dup, dup_set


def _allocate_patch(indices: np.ndarray, overlap: float, point_set: Set[int], duplicates_set: Set[int]):
    overlap_indices = int(indices.shape[0] * overlap)
    if duplicates_set:
        for raw_idx in indices[1:overlap_indices]:
            idx = int(raw_idx)
            if idx not in duplicates_set:
                point_set.discard(idx)
    else:
        for raw_idx in indices[1:overlap_indices]:
            point_set.discard(int(raw_idx))
    return point_set


def closest_point(index: int, included: Set[int], points: np.ndarray):
    if not included:
        return index, 0.0
    p = points[index]
    dim = points.shape[1]
    min_distance_sq = 0.0
    closest_index = index
    for ii in included:
        row = points[ii]
        distance_sq = 0.0
        for jj in range(dim):
            diff = row[jj] - p[jj]
            distance_sq += diff * diff
        if min_distance_sq == 0.0:
            if distance_sq > 0.0:
                min_distance_sq = distance_sq
                closest_index = int(ii)
            elif distance_sq == 0.0:
                closest_index = int(ii)
        elif distance_sq == min_distance_sq:
            closest_index = int(ii)
        elif distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest_index = int(ii)
    if min_distance_sq == 0.0:
        return closest_index, 0.0
    return closest_index, math.sqrt(min_distance_sq)


def get_angle(index_1: int, index_2: int, normals: np.ndarray) -> float:
    v1 = normals[index_1]
    v2 = normals[index_2]
    dim = normals.shape[1]
    dot = 0.0
    n1_sq = 0.0
    n2_sq = 0.0
    for jj in range(dim):
        a = float(v1[jj])
        b = float(v2[jj])
        dot += a * b
        n1_sq += a * a
        n2_sq += b * b
    if n1_sq == 0.0 or n2_sq == 0.0:
        return 0.0
    ratio = dot / math.sqrt(n1_sq * n2_sq)
    if ratio < -1.0:
        ratio = -1.0
    elif ratio > 1.0:
        ratio = 1.0
    return math.acos(ratio) * _RAD_TO_DEG


def _allocate_angle(idx: int, indices: np.ndarray, points: np.ndarray, normals: np.ndarray, feature_angle: float):
    include_points: Set[int] = set()
    allocated: List[int] = []
    n_points = points.shape[0]
    point_dim = points.shape[1]
    normal_dim = normals.shape[1]
    seed_idx = int(idx)
    for ii in range(n_points):
        index_value = int(indices[ii])
        if index_value == seed_idx:
            include_points.add(ii)
            allocated.append(index_value)
            break
    for ii in range(1, n_points):
        if not include_points:
            continue
        p = points[ii]
        min_distance_sq = 0.0
        closest_idx = ii
        for jj in include_points:
            row = points[jj]
            distance_sq = 0.0
            for kk in range(point_dim):
                diff = row[kk] - p[kk]
                distance_sq += diff * diff
            if min_distance_sq == 0.0:
                if distance_sq > 0.0:
                    min_distance_sq = distance_sq
                    closest_idx = jj
                elif distance_sq == 0.0:
                    closest_idx = jj
            elif distance_sq == min_distance_sq:
                closest_idx = jj
            elif distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_idx = jj
        if min_distance_sq == 0.0:
            continue
        v1 = normals[ii]
        v2 = normals[closest_idx]
        dot = 0.0
        n1_sq = 0.0
        n2_sq = 0.0
        for kk in range(normal_dim):
            a = float(v1[kk])
            b = float(v2[kk])
            dot += a * b
            n1_sq += a * a
            n2_sq += b * b
        if n1_sq == 0.0 or n2_sq == 0.0:
            angle = 0.0
        else:
            ratio = dot / math.sqrt(n1_sq * n2_sq)
            if ratio < -1.0:
                ratio = -1.0
            elif ratio > 1.0:
                ratio = 1.0
            angle = math.acos(ratio) * _RAD_TO_DEG
        if angle > feature_angle:
            continue
        include_points.add(ii)
        allocated.append(int(indices[ii]))
    return allocated
