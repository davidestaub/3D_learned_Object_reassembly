"""
Geometry utilities useful in dataloaders.
"""

import numpy as np
from pathlib import Path

from .colmap.read_model import read_model, qvec2rotmat

exif_str_to_rot = {
    'TopLeft': 0, 'BottomRight': 2, 'RightTop': 3, 'LeftBottom': 1,
    'Undefined': 0,
}

rotation_matrices = [
    np.array([[np.cos(r), -np.sin(r), 0., 0.],
              [np.sin(r), np.cos(r), 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]], dtype=np.float32)
    for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
]


def intrinsics_and_extrinsics_from_model(args):
    """Extract poses and intrinsics from a COLAMP model."""
    path, ids = args
    cameras, images, _ = read_model(
        path, ext='.bin', read_points=False)
    intrinsics, poses = {}, {}
    for i in ids:
        fx, fy, cx, cy = cameras[images[i].camera_id].params
        intrinsics[i] = np.array([[fx, 0, cx-0.5],
                                  [0, fy, cy-0.5],
                                  [0, 0, 1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = qvec2rotmat(images[i].qvec)
        T[:3, 3] = images[i].tvec
        poses[i] = T  # pose is T_world_to_image
    return intrinsics, poses


def rotation_from_exif(exif_path):
    """Extractor the in-plane rotation from an exif file."""
    if Path(exif_path).exists():
        with open(exif_path, 'r', errors='ignore') as f:
            for line in f:
                if '  Orientation:' in line:
                    return exif_str_to_rot[line.split()[1]]
    return None


def rotate_intrinsics(K, image_shape, rot):
    """Correct the intrinsics after in-plane rotation.
    Args:
        K: the original (3, 3) intrinsic matrix.
        image_shape: shape of the image after rotation `[H, W]`.
        rot: the number of clockwise 90deg rotations.
    """
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 0:
        return K
    elif rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)
    else:
        raise ValueError


def scale_intrinsics(K, scales):
    """Scale intrinsics after resizing the corresponding image."""
    scales = np.diag(np.concatenate([scales, [1.]]))
    return np.dot(scales.astype(K.dtype, copy=False), K)


def rotate_pose_inplane(i_T_w, rot):
    """Apply an in-plane rotation to a pose (world to camera)."""
    return np.dot(rotation_matrices[rot], i_T_w)
