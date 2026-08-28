import numpy as np

from svv.tree.data.data import TrackedIndexList, serialize_tree_map
from svv.tree.tree import Tree


def test_tree_growth_work_caches_match_preallocation():
    tree = Tree(preallocation_step=4)

    assert np.array_equal(tree._row_idx_cache, np.arange(4, dtype=np.intp))
    assert np.array_equal(
        tree._col_idx_cache,
        np.arange(tree.preallocate.shape[1], dtype=np.intp),
    )
    assert tree._radius_work.shape == (4,)
    assert tree._tmp_28_work.shape == (4,)
    assert tree._radius_work.dtype == tree.preallocate.dtype
    assert tree._tmp_28_work.dtype == tree.preallocate.dtype
    assert tree._change_i_work.shape == (4,)
    assert tree._change_j_work.shape == (4,)
    assert tree._new_data_work.shape == (4,)
    assert tree._old_data_work.shape == (4,)
    assert np.array_equal(tree._main_update_cols, np.array([22, 25, 27, 28, 23, 24], dtype=np.intp))


def test_tree_growth_work_caches_refresh_on_resize():
    tree = Tree(preallocation_step=2)
    old_row_cache = tree._row_idx_cache
    old_col_cache = tree._col_idx_cache
    old_radius_work = tree._radius_work
    old_tmp_28_work = tree._tmp_28_work

    tree.ensure_preallocation(5)

    capacity = tree.preallocate.shape[0]
    assert capacity >= 5
    assert tree._row_idx_cache.shape == (capacity,)
    assert np.array_equal(tree._row_idx_cache[:5], np.arange(5, dtype=np.intp))
    assert tree._col_idx_cache.shape == (tree.preallocate.shape[1],)
    assert np.array_equal(
        tree._col_idx_cache,
        np.arange(tree.preallocate.shape[1], dtype=np.intp),
    )
    assert tree._radius_work.shape == (capacity,)
    assert tree._tmp_28_work.shape == (capacity,)
    assert tree._radius_work.dtype == tree.preallocate.dtype
    assert tree._tmp_28_work.dtype == tree.preallocate.dtype
    assert tree._row_idx_cache is not old_row_cache
    assert tree._col_idx_cache is not old_col_cache
    assert tree._radius_work is not old_radius_work
    assert tree._tmp_28_work is not old_tmp_28_work


def test_tree_change_work_caches_refresh_on_demand():
    tree = Tree(preallocation_step=2)
    old_change_i = tree._change_i_work
    old_change_j = tree._change_j_work
    old_new_data = tree._new_data_work
    old_old_data = tree._old_data_work

    tree._ensure_change_work_capacity(7, np.float32)

    assert tree._change_i_work.shape == (7,)
    assert tree._change_j_work.shape == (7,)
    assert tree._new_data_work.shape == (7,)
    assert tree._old_data_work.shape == (7,)
    assert tree._new_data_work.dtype == np.float32
    assert tree._old_data_work.dtype == np.float32
    assert tree._change_i_work is not old_change_i
    assert tree._change_j_work is not old_change_j
    assert tree._new_data_work is not old_new_data
    assert tree._old_data_work is not old_old_data


def test_tree_downstream_cache_refreshes_after_direct_mutation():
    tree = Tree(preallocation_step=2)
    tree.vessel_map = {
        0: {"upstream": [], "downstream": [1, 2]},
        1: {"upstream": [0], "downstream": []},
        2: {"upstream": [0], "downstream": []},
    }

    assert isinstance(tree.vessel_map[0]["downstream"], TrackedIndexList)
    assert np.array_equal(tree.get_downstream_indices(0), np.array([1, 2], dtype=np.intp))
    assert tree.get_downstream_count(0) == 2

    tree.vessel_map[0]["downstream"].append(3)

    assert np.array_equal(tree.get_downstream_indices(0), np.array([1, 2, 3], dtype=np.intp))
    assert tree.get_downstream_count(0) == 3

    tree.vessel_map[0]["downstream"] = [4, 5]

    assert np.array_equal(tree.get_downstream_indices(0), np.array([4, 5], dtype=np.intp))
    assert tree.get_downstream_count(0) == 2


def test_tree_downstream_cache_refreshes_after_map_replacement_and_serializes_plain_lists():
    tree = Tree(preallocation_step=2)
    tree.vessel_map = {7: {"upstream": [], "downstream": [8, 9]}}

    assert np.array_equal(tree.get_downstream_indices(7), np.array([8, 9], dtype=np.intp))
    assert tree.get_downstream_count(7) == 2

    serialized = serialize_tree_map(tree.vessel_map)

    assert serialized == {7: {"upstream": [], "downstream": [8, 9]}}
    assert isinstance(serialized[7]["downstream"], list)
    assert not isinstance(serialized[7]["downstream"], TrackedIndexList)


def test_tree_sum_downstream_counts_counts_duplicates_and_refreshes_dirty_entries():
    tree = Tree(preallocation_step=4)
    tree.vessel_map = {
        0: {"upstream": [], "downstream": [1, 2]},
        1: {"upstream": [0], "downstream": [3]},
        2: {"upstream": [0], "downstream": []},
        3: {"upstream": [1], "downstream": []},
    }

    assert tree.sum_downstream_counts(np.array([0, 1, 0, -1], dtype=np.intp)) == 5

    tree.vessel_map[1]["downstream"].append(4)

    assert tree.sum_downstream_counts(np.array([1, 0], dtype=np.intp)) == 4
