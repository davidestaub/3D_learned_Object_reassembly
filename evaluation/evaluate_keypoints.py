import argparse
import copy
import functools
import math
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import open3d as o3d
import pandas as pd
from open3d.cpu.pybind.geometry import PointCloud
from scipy.spatial.distance import cdist
from tqdm import tqdm

from process_folder_cluster import get_keypoints

np.random.seed(42)

# A random rotation.
angles = np.random.uniform(low=0, high=2 * math.pi, size=(3,))
ROTATION = PointCloud().get_rotation_matrix_from_xyz(angles)
REVERSE_ROTATION = PointCloud().get_rotation_matrix_from_xyz(-angles)
KEYPOINT_METHODS = ['SD', 'sticky', 'hybrid']  # , 'harris']
NUM_KEYPOINTS = 512


def calculate_score(keypoints1, keypoints2):
    dists = cdist(keypoints1, keypoints2)
    closest = np.concatenate([np.min(dists, axis=0), np.min(dists, axis=1)])
    return closest


def evaluate_repeatability(fragment_idx, pcd, args, method):
    c = pcd.get_center()
    pcd = pcd.translate(-1 * c)
    pcd_rotated = copy.deepcopy(pcd).rotate(ROTATION, center=(0, 0, 0))

    args.keypoint_method = method
    keypoints = get_keypoints(fragment_idx, np.array(pcd.points), np.array(pcd.normals),
                              args=args, desc_normal=None, desc_inv=None, folder_path='', npoints=NUM_KEYPOINTS,
                              save=False)
    keypoints_rotated = get_keypoints(fragment_idx, np.array(pcd_rotated.points), np.array(pcd_rotated.normals),
                                      args=args, desc_normal=None, desc_inv=None, folder_path='', npoints=NUM_KEYPOINTS,
                                      save=False)

    kpts_pcd = PointCloud(o3d.utility.Vector3dVector(keypoints[:, :3]))
    kpts_pcd_rotated = PointCloud(o3d.utility.Vector3dVector(keypoints_rotated[:, :3]))
    kpts_pcd_rotated = kpts_pcd_rotated.rotate(REVERSE_ROTATION)

    closest = calculate_score(kpts_pcd.points, kpts_pcd_rotated.points)

    return closest


def process_folder(args, folder_path):
    object_name = os.path.basename(folder_path)
    obj_files = glob(os.path.join(args.path, folder_path, 'cleaned', '*.pcd'))

    num_fragments = len(obj_files)
    if num_fragments == 0:
        print(f"No fragments found in {folder_path}")
        return []

    fragment_pcds = []
    for i in range(num_fragments):
        file_path = os.path.join(args.path, folder_path, 'cleaned', f'{object_name}_cleaned.{i}.pcd')
        fragment_pcds.append(o3d.io.read_point_cloud(file_path))

    # object name, fragment, method, mean, std
    data = []
    for i, pcd in enumerate(fragment_pcds):
        for method in KEYPOINT_METHODS:
            distances = evaluate_repeatability(i, pcd, args, method)
            data.append((object_name, i, method, *distances))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate repeatability and different fragment matching of keypoints."
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--keypoint_method", type=str, default='hybrid', choices=['SD', 'sticky', 'hybrid'])
    parser.add_argument("--descriptor_method", type=str, default='fpfh', choices=['fpfh', 'pillar', 'fpfh_pillar'])
    args = parser.parse_args()

    paths = [os.path.abspath(os.path.join(args.path, folder)) for folder in os.listdir(args.path)]
    fn = functools.partial(process_folder, args)
    with Pool() as p:
        data = list(tqdm(p.imap(fn, paths), total=len(paths)))

    df = pd.DataFrame(data=[entry for sublist in data for entry in sublist],
                      columns=['object_name', 'fragment_idx', 'method', *[f'k_{i}' for i in range(2 * NUM_KEYPOINTS)]])
    df.to_csv('keypoint_repeatability.csv')
    print(df.sort_values(by='method'))
