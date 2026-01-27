# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for working with eyebrow keypoints in the ava-256 dataset.

Provides functions to:
- Transform 3D keypoints into UV texture coordinates
- Project coordinates/segmentation masks from UV space into 2D camera space
- Extract eyebrow-specific keypoints (indices 8-25)
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from trimesh import Trimesh
from trimesh.triangles import points_to_barycentric
from zipp import Path as ZipPath

from utils import closest_point_barycentrics, load_camera_calibration, load_obj


# Eyebrow keypoint indices in the ava-256 dataset
EYEBROW_START_IDX = 8
EYEBROW_END_IDX = 25  # inclusive


def load_keypoints_3d(base_dir: Union[str, Path], frame_id: int) -> np.ndarray:
    """
    Load 3D keypoints from the dataset.

    Args:
        base_dir: Path to the decoder directory (e.g., "{dataset}/{subject_id}/decoder")
        frame_id: Frame number to load

    Returns:
        np.ndarray: Keypoints array of shape (N, 6) where columns are:
            [keypoint_idx, x, y, z, confidence_sum, num_inliers]
    """
    path = ZipPath(f"{base_dir}/keypoints_3d/keypoints_3d.zip", f"{frame_id:06d}.npy")
    keypoints = np.load(io.BytesIO(path.read_bytes()))
    return keypoints.reshape(-1, 6)


def load_head_pose(base_dir: Union[str, Path], frame_id: int) -> np.ndarray:
    """
    Load head pose transformation matrix.

    Args:
        base_dir: Path to the decoder directory
        frame_id: Frame number to load

    Returns:
        np.ndarray: 3x4 transformation matrix
    """
    path = ZipPath(f"{base_dir}/head_pose/head_pose.zip", f"{frame_id:06d}.txt")
    head_pose = np.loadtxt(io.BytesIO(path.read_bytes()), dtype=np.float32)
    return head_pose


def get_eyebrow_keypoints(
    keypoints_3d: np.ndarray,
    left: bool = True,
    right: bool = True,
) -> np.ndarray:
    """
    Extract eyebrow keypoints from the full keypoints array.

    Eyebrow keypoints are at indices 8-25 in the ava-256 dataset.

    Args:
        keypoints_3d: Full keypoints array of shape (N, 6)
        left: Include left eyebrow keypoints
        right: Include right eyebrow keypoints

    Returns:
        np.ndarray: Filtered eyebrow keypoints of shape (M, 6)
    """
    # Filter to only keypoints with indices in the eyebrow range
    mask = (keypoints_3d[:, 0] >= EYEBROW_START_IDX) & (keypoints_3d[:, 0] <= EYEBROW_END_IDX)
    eyebrow_kpts = keypoints_3d[mask]

    # Note: Left/right filtering would require knowing the exact split point
    # For now, return all eyebrow keypoints
    # TODO: Add left/right split if spatial information is available
    if not (left and right):
        # Split roughly in half based on keypoint index
        mid_idx = (EYEBROW_START_IDX + EYEBROW_END_IDX) // 2
        if left and not right:
            eyebrow_kpts = eyebrow_kpts[eyebrow_kpts[:, 0] <= mid_idx]
        elif right and not left:
            eyebrow_kpts = eyebrow_kpts[eyebrow_kpts[:, 0] > mid_idx]

    return eyebrow_kpts


def project_points_3d_to_2d(
    points_3d: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    head_pose: Optional[np.ndarray] = None,
    image_scale: float = 4.0,
) -> np.ndarray:
    """
    Project 3D points to 2D camera coordinates.

    Follows the projection pattern from demos/keypoints.py:
    1. Apply head pose transformation (if provided)
    2. Project using camera intrinsic/extrinsic matrices
    3. Perspective division
    4. Scale for downsampled images

    Args:
        points_3d: 3D points of shape (N, 3)
        camera_params: Dictionary with 'intrin' and 'extrin' matrices
        head_pose: Optional 3x4 head pose transformation matrix
        image_scale: Scale factor for downsampled images (default 4.0)

    Returns:
        np.ndarray: 2D points of shape (2, N) as [x_coords, y_coords]
    """
    intrin = camera_params["intrin"]
    extrin = camera_params["extrin"]

    # Ensure points are (N, 3)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)

    n_points = points_3d.shape[0]

    # Add homogeneous coordinate
    points_h = np.hstack([points_3d, np.ones((n_points, 1))])  # (N, 4)

    # Apply head pose if provided
    if head_pose is not None:
        # head_pose is 3x4, points_h is (N, 4)
        points_world = (head_pose @ points_h.T)  # (3, N)
        # Add homogeneous coordinate for camera projection
        points_world = np.vstack([points_world, np.ones((1, n_points))])  # (4, N)
    else:
        points_world = points_h.T  # (4, N)

    # Project using camera matrices
    twod = intrin @ extrin @ points_world  # (3, N)

    # Perspective division
    twod = twod / twod[2:3, :]

    # Scale for downsampled images
    twod = twod / image_scale

    return twod[:2, :]  # (2, N)


def keypoints_3d_to_uv(
    keypoints_3d: np.ndarray,
    mesh_v: np.ndarray,
    mesh_vi: np.ndarray,
    mesh_vt: np.ndarray,
    mesh_vti: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D keypoints into UV texture coordinates.

    Uses barycentric interpolation to find the UV coordinates of the closest
    point on the mesh surface for each 3D keypoint.

    Args:
        keypoints_3d: 3D keypoints of shape (N, 3) - just the xyz coordinates
        mesh_v: Mesh vertices of shape (V, 3)
        mesh_vi: Mesh face vertex indices of shape (F, 3)
        mesh_vt: Mesh texture coordinates of shape (T, 2)
        mesh_vti: Mesh face texture coordinate indices of shape (F, 3)

    Returns:
        np.ndarray: UV coordinates of shape (N, 2)
    """
    # Find closest points on mesh and get barycentric coordinates
    _, barys, _, face_idxs = closest_point_barycentrics(mesh_v, mesh_vi, keypoints_3d)

    # Get texture coordinate indices for each face
    vti_faces = mesh_vti[face_idxs]  # (N, 3)

    # Get texture coordinates for each vertex of the triangle
    uv0 = mesh_vt[vti_faces[:, 0]]  # (N, 2)
    uv1 = mesh_vt[vti_faces[:, 1]]
    uv2 = mesh_vt[vti_faces[:, 2]]

    # Interpolate UV using barycentric coordinates
    b0, b1, b2 = barys[:, 0:1], barys[:, 1:2], barys[:, 2:3]
    uv_coords = b0 * uv0 + b1 * uv1 + b2 * uv2

    return uv_coords


def uv_to_3d(
    uv_coords: np.ndarray,
    mesh_v: np.ndarray,
    mesh_vi: np.ndarray,
    mesh_vt: np.ndarray,
    mesh_vti: np.ndarray,
) -> np.ndarray:
    """
    Convert UV coordinates to 3D positions on the mesh surface.

    Uses barycentric interpolation in UV space to find corresponding 3D positions.

    Args:
        uv_coords: UV coordinates of shape (N, 2)
        mesh_v: Mesh vertices of shape (V, 3)
        mesh_vi: Mesh face vertex indices of shape (F, 3)
        mesh_vt: Mesh texture coordinates of shape (T, 2)
        mesh_vti: Mesh face texture coordinate indices of shape (F, 3)

    Returns:
        np.ndarray: 3D positions of shape (N, 3)
    """
    # Create a 2D "mesh" in UV space (add z=0)
    vt_3d = np.hstack([mesh_vt, np.zeros((mesh_vt.shape[0], 1))])
    uv_3d = np.hstack([uv_coords, np.zeros((uv_coords.shape[0], 1))])

    # Find closest points in UV space
    _, barys, _, face_idxs = closest_point_barycentrics(vt_3d, mesh_vti, uv_3d)

    # Get 3D vertex indices for the corresponding faces
    vi_faces = mesh_vi[face_idxs]  # (N, 3)

    # Get 3D vertex positions
    v0 = mesh_v[vi_faces[:, 0]]  # (N, 3)
    v1 = mesh_v[vi_faces[:, 1]]
    v2 = mesh_v[vi_faces[:, 2]]

    # Interpolate 3D position using barycentric coordinates
    b0, b1, b2 = barys[:, 0:1], barys[:, 1:2], barys[:, 2:3]
    positions_3d = b0 * v0 + b1 * v1 + b2 * v2

    return positions_3d


def project_uv_to_2d(
    uv_coords: np.ndarray,
    mesh_v: np.ndarray,
    mesh_vi: np.ndarray,
    mesh_vt: np.ndarray,
    mesh_vti: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    head_pose: Optional[np.ndarray] = None,
    image_scale: float = 4.0,
) -> np.ndarray:
    """
    Project UV coordinates to 2D camera coordinates.

    First converts UV to 3D positions on the mesh, then projects to 2D.

    Args:
        uv_coords: UV coordinates of shape (N, 2)
        mesh_v: Mesh vertices of shape (V, 3)
        mesh_vi: Mesh face vertex indices of shape (F, 3)
        mesh_vt: Mesh texture coordinates of shape (T, 2)
        mesh_vti: Mesh face texture coordinate indices of shape (F, 3)
        camera_params: Dictionary with 'intrin' and 'extrin' matrices
        head_pose: Optional 3x4 head pose transformation matrix
        image_scale: Scale factor for downsampled images

    Returns:
        np.ndarray: 2D points of shape (2, N)
    """
    # Convert UV to 3D
    points_3d = uv_to_3d(uv_coords, mesh_v, mesh_vi, mesh_vt, mesh_vti)

    # Project to 2D
    return project_points_3d_to_2d(points_3d, camera_params, head_pose, image_scale)


def project_segmentation_from_uv(
    uv_mask: np.ndarray,
    mesh_v: np.ndarray,
    mesh_vi: np.ndarray,
    mesh_vt: np.ndarray,
    mesh_vti: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    head_pose: np.ndarray,
    output_size: Tuple[int, int],
    image_scale: float = 4.0,
) -> np.ndarray:
    """
    Project a UV-space segmentation mask to 2D camera space.

    Rasterizes UV mask pixels through mesh triangles to camera view.

    Args:
        uv_mask: Binary UV mask of shape (H, W)
        mesh_v: Mesh vertices of shape (V, 3)
        mesh_vi: Mesh face vertex indices of shape (F, 3)
        mesh_vt: Mesh texture coordinates of shape (T, 2)
        mesh_vti: Mesh face texture coordinate indices of shape (F, 3)
        camera_params: Dictionary with 'intrin' and 'extrin' matrices
        head_pose: 3x4 head pose transformation matrix
        output_size: (height, width) of output mask
        image_scale: Scale factor for downsampled images

    Returns:
        np.ndarray: Binary mask of shape output_size
    """
    h_uv, w_uv = uv_mask.shape
    h_out, w_out = output_size

    # Find all non-zero pixels in UV mask
    v_indices, u_indices = np.where(uv_mask > 0)

    if len(u_indices) == 0:
        return np.zeros(output_size, dtype=np.uint8)

    # Convert pixel coordinates to UV coordinates [0, 1]
    # UV origin is typically bottom-left in OpenGL convention, but we use top-left
    uv_coords = np.stack([
        (u_indices + 0.5) / w_uv,
        1.0 - (v_indices + 0.5) / h_uv,
    ], axis=1)

    # Project UV coordinates to 2D
    points_2d = project_uv_to_2d(
        uv_coords, mesh_v, mesh_vi, mesh_vt, mesh_vti,
        camera_params, head_pose, image_scale
    )

    # Create output mask
    output_mask = np.zeros(output_size, dtype=np.uint8)

    # Rasterize points to output mask
    x_coords = np.round(points_2d[0]).astype(int)
    y_coords = np.round(points_2d[1]).astype(int)

    # Filter valid coordinates
    valid = (x_coords >= 0) & (x_coords < w_out) & (y_coords >= 0) & (y_coords < h_out)
    output_mask[y_coords[valid], x_coords[valid]] = 1

    return output_mask


def get_eyebrow_keypoints_2d(
    base_dir: Union[str, Path],
    frame_id: int,
    camera_id: str,
    topology_path: str = "./assets/face_topology.obj",
    image_scale: float = 4.0,
) -> np.ndarray:
    """
    High-level convenience function to get 2D eyebrow keypoint positions.

    Loads all required data and returns projected 2D keypoint positions.

    Args:
        base_dir: Path to the decoder directory
        frame_id: Frame number
        camera_id: Camera identifier (e.g., "400191")
        topology_path: Path to face topology OBJ file
        image_scale: Scale factor for downsampled images

    Returns:
        np.ndarray: 2D keypoint positions of shape (2, N) as [x_coords, y_coords]
    """
    # Load keypoints
    keypoints = load_keypoints_3d(base_dir, frame_id)
    eyebrow_kpts = get_eyebrow_keypoints(keypoints)

    if len(eyebrow_kpts) == 0:
        return np.array([[], []])

    # Extract 3D coordinates (columns 1-3)
    points_3d = eyebrow_kpts[:, 1:4]

    # Load head pose
    head_pose = load_head_pose(base_dir, frame_id)

    # Load camera calibration
    camera_calibration = load_camera_calibration(f"{base_dir}/camera_calibration.json")
    camera_params = camera_calibration[camera_id]

    # Project to 2D
    return project_points_3d_to_2d(points_3d, camera_params, head_pose, image_scale)


def visualize_keypoints_on_image(
    image: Image.Image,
    keypoints_2d: np.ndarray,
    color: str = "red",
    size: int = 5,
) -> Image.Image:
    """
    Helper function to visualize keypoints on an image.

    Args:
        image: PIL Image to draw on
        keypoints_2d: 2D keypoints of shape (2, N)
        color: Color for keypoint markers
        size: Size of keypoint markers

    Returns:
        Image with keypoints drawn
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    ax.scatter(keypoints_2d[0], keypoints_2d[1], c=color, s=size)
    ax.axis('off')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    result = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close(fig)

    return result
