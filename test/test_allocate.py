import pytest
import numpy as np

from svv.domain.domain import Domain
from svv.domain.routines.allocate import allocate

@pytest.fixture
def simple_points():
    """
    A small set of 2D or 3D points.
    We’ll just do a simple 2D example for convenience
    that can be extended to 3D.
    """
    # Example: four points forming a square in 2D (z=0)
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    return pts


@pytest.fixture
def simple_points_and_normals(simple_points):
    """
    Provide a matching array of normals for each point.
    """
    # Let’s just say each normal is pointing 'up' in z direction in 3D
    # or if using 2D arrays, we can store Nx2 for 2D "normals".
    # For demonstration, let's assume 3D with z=1 for all:
    pts = simple_points
    # Convert to 3D by appending z=0
    pts_3d = np.hstack([pts, np.zeros((pts.shape[0], 1))])
    # Normals all point in +z
    normals_3d = np.zeros_like(pts_3d)
    normals_3d[:, 2] = 1.0
    return pts_3d, normals_3d


def test_allocate_no_data():
    """
    Test calling allocate with no arguments,
    expecting it to return [None, None] and print an error message.
    """
    result = allocate()
    assert result == [None, None], "allocate() with no data should return [None, None]"


def test_allocate_only_points(simple_points):
    """
    Test calling allocate with only points, no normals.
    """
    patches = allocate(simple_points)
    # Check if we get something valid back
    assert isinstance(patches, list), "Expected a list of patches."
    # Each element of patches is (patch_points, patch_normals)
    for p_pts, p_normals in patches:
        assert p_normals is None, "No normals given => patch_normals should be None."
        assert p_pts.shape[1] == simple_points.shape[1], "Patch points should match dimension of input."


def test_allocate_points_and_normals(simple_points_and_normals):
    """
    Test calling allocate with points and normals.
    """
    pts, normals = simple_points_and_normals
    patches = allocate(pts, normals)
    assert isinstance(patches, list), "Expected a list of patches."
    for p_pts, p_normals in patches:
        # Each patch’s dimension should be Nx3
        # (since we used 3D points and normals).
        if p_pts is not None:
            assert p_pts.shape[1] == 3, "Patch points should have 3D coords."
        if p_normals is not None:
            assert p_normals.shape[1] == 3, "Patch normals should have 3D coords."


def test_allocate_zero_magnitude_normals():
    """
    Provide a few points with zero-magnitude normals to ensure
    the function logs an error and removes them.
    """
    pts = np.array([
        [0,0,0],
        [1,0,0],
        [2,0,0],
    ], dtype=float)
    # Normals: second is zero
    normals = np.array([
        [0,0,1],
        [0,0,0],  # zero magnitude
        [0,0,1],
    ], dtype=float)
    patches = allocate(pts, normals)
    # The function is supposed to remove the point with zero normal
    # so only 2 points remain
    # Typically the function prints a message about zero magnitude normals
    # We can check the patch data:
    all_patch_points = np.vstack([pp for pp,_ in patches])
    # Because patches may overlap
    unique_pts_in_patches = np.unique(all_patch_points, axis=0)
    assert len(unique_pts_in_patches) == 2, (
        "Points with zero normals should have been removed, leaving 2 points."
    )


def test_allocate_duplicates():
    """
    Test handling of exact duplicate points.
    If duplicates exist, the code might remove or
    handle them in a special way.
    """
    pts = np.array([
        [0,0,0],
        [1,1,1],
        [1,1,1],  # duplicate
        [2,2,2],
    ], dtype=float)
    # Let’s say all normals are (0,0,1) to keep it simple
    normals = np.array([
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
    ], dtype=float)
    patches = allocate(pts, normals)
    # Check that it runs and returns patches
    assert isinstance(patches, list), "Expected a list of patches even with duplicates."

    # See how many total points remain in the patches
    # The code might remove duplicates if they have identical normals
    # or handle them in a certain way.
    # For a simple check, we’ll ensure we have <= 4 points total
    all_pts_in_patches = np.vstack([pp for pp,_ in patches])
    unique_pts_in_patches = np.unique(all_pts_in_patches, axis=0)
    # The function may remove or keep duplicates depending on logic.
    # If it removes duplicates, we expect 3 or fewer. If not, we might see 4.
    # We can't say exactly unless we know the intended logic.
    # But let's ensure it doesn't crash.
    assert len(unique_pts_in_patches) <= 4, "No crash or unexpected addition of points."


def test_allocate_overlap_and_angle(simple_points_and_normals):
    """
    Test with custom overlap and feature_angle settings.
    Ensures function can handle them without error.
    """
    pts, normals = simple_points_and_normals
    patches = allocate(pts, normals, min_patch_size=2, max_patch_size=4,
                       overlap=0.5, feature_angle=45)
    # We just test that it doesn’t crash and returns a list
    assert isinstance(patches, list)
    # Optionally, we can check how many patches we get,
    # or that each patch has at least 2 points, etc.
    for p_pts, p_norms in patches:
        if p_pts is not None:
            assert len(p_pts) >= 2, "min_patch_size=2 => each patch should have >=2 points."


def test_allocate_large_data():
    """
    Stress test with a larger random point cloud
    (without normals) to ensure performance logic at least runs.
    """
    np.random.seed(42)
    pts = np.random.rand(500, 3)  # 500 random points in 3D
    patches = allocate(pts, min_patch_size=5, max_patch_size=20, overlap=0.1)
    assert isinstance(patches, list), "Expected a list of patches."
    # Just ensure it doesn't fail for bigger data.


def test_allocate_invalid_overlap(simple_points):
    """
    Overlap outside [0,1] gets clipped.
    Ensure the function doesn't crash or do something weird.
    """
    # Overlap = 2 => clipped to 1
    patches = allocate(simple_points, min_patch_size=2, max_patch_size=3, overlap=2)
    assert isinstance(patches, list), "Should still return a list."
    # Overlap = -0.5 => clipped to 0
    patches = allocate(simple_points, min_patch_size=2, max_patch_size=3, overlap=-0.5)
    assert isinstance(patches, list), "Should still return a list."


def test_allocate_progress_callback_is_monotonic():
    """
    Progress callbacks should report monotonic normalized progress and finish at 1.0.
    """
    rng = np.random.default_rng(42)
    pts = rng.random((64, 3))
    updates = []

    def on_progress(progress, label=None, indeterminate=None):
        updates.append((progress, label, indeterminate))

    patches = allocate(
        pts,
        min_patch_size=5,
        max_patch_size=12,
        overlap=0.1,
        progress_callback=on_progress,
    )

    assert isinstance(patches, list)
    progress_values = [progress for progress, _, _ in updates if progress is not None]
    assert progress_values, "Expected at least one progress update."
    assert progress_values[0] == pytest.approx(0.0)
    assert progress_values[-1] == pytest.approx(1.0)
    assert all(
        b >= a for a, b in zip(progress_values, progress_values[1:])
    ), "Progress callback values should be monotonic."
    assert any(
        label == "Allocating patches (estimating remaining seeds)..."
        for _, label, _ in updates
    ), "Expected adaptive seed-estimate progress messages from allocate()."


def test_domain_create_progress_callback_reaches_completion(simple_points):
    """
    Domain.create should surface normalized progress from allocation through patch creation.
    """
    domain = Domain(simple_points)
    updates = []

    def on_progress(progress, label=None, indeterminate=None):
        updates.append((progress, label, indeterminate))

    domain.create(
        min_patch_size=2,
        max_patch_size=3,
        overlap=0.2,
        progress_callback=on_progress,
    )

    assert len(domain.patches) > 0
    progress_values = [progress for progress, _, _ in updates if progress is not None]
    assert progress_values[0] == pytest.approx(0.0)
    assert progress_values[-1] == pytest.approx(1.0)
    assert all(
        b >= a for a, b in zip(progress_values, progress_values[1:])
    ), "Domain.create progress should be monotonic."
