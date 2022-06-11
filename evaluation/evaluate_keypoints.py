import argparse
import copy
import functools
import math
import os
from glob import glob
from typing import Dict, List

import numpy as np
import open3d as o3d
import pandas as pd
from open3d.cpu.pybind.geometry import PointCloud
from scipy.spatial import KDTree

from object_fracturing.process_data import get_keypoints, get_fragment_matchings

np.random.seed(42)

# A random rotation.
angles = np.random.uniform(low=0, high=2 * math.pi, size=(3,))
ROTATION = PointCloud().get_rotation_matrix_from_xyz(angles)
REVERSE_ROTATION = PointCloud().get_rotation_matrix_from_xyz(-angles)

angles_2 = np.random.uniform(low=0, high=2 * math.pi, size=(3,))
ROTATION_2 = PointCloud().get_rotation_matrix_from_xyz(angles_2)
REVERSE_ROTATION_2 = PointCloud().get_rotation_matrix_from_xyz(-angles_2)

KEYPOINT_METHODS = ['SD', 'sticky', 'hybrid', 'iss']  # , 'harris']
# KEYPOINT_METHODS = ['harris']
NUM_KEYPOINTS = 512


def calculate_score(keypoints1, keypoints2):
    tree = KDTree(keypoints1)
    dists_to_closest1, _ = tree.query(keypoints2)
    tree = KDTree(keypoints2)
    dists_to_closest2, _ = tree.query(keypoints1)
    closest = np.concatenate([dists_to_closest1, dists_to_closest2])
    return closest


def evaluate_repeatability(fragment_idx, pcd, args, method, folder_path):
    c = pcd.get_center()
    pcd = pcd.translate(-1 * c)
    pcd_rotated = copy.deepcopy(pcd).rotate(ROTATION, center=(0, 0, 0))
    pcd_rotated_2 = copy.deepcopy(pcd).rotate(ROTATION_2, center=(0, 0, 0))

    args.keypoint_method = method
    keypoints_rotated, _ = get_keypoints(fragment_idx, np.array(pcd_rotated.points), np.array(pcd_rotated.normals),
                                      method=method, folder_path=folder_path,
                                      npoints=NUM_KEYPOINTS, tag='rotated')
    keypoints_rotated_2, _ = get_keypoints(fragment_idx, np.array(pcd_rotated_2.points), np.array(pcd_rotated_2.normals),
                                        method=method, folder_path=folder_path,
                                        npoints=NUM_KEYPOINTS, tag='rotated_2')
    kpts_pcd_rotated = PointCloud(o3d.utility.Vector3dVector(keypoints_rotated[:, :3]))
    kpts_pcd_rotated = kpts_pcd_rotated.rotate(REVERSE_ROTATION)

    kpts_pcd_rotated_2 = PointCloud(o3d.utility.Vector3dVector(keypoints_rotated_2[:, :3]))
    kpts_pcd_rotated_2 = kpts_pcd_rotated_2.rotate(REVERSE_ROTATION_2)

    closest = calculate_score(kpts_pcd_rotated_2.points, kpts_pcd_rotated.points)

    return closest


def evaluate_surface_repeatability(fragments: List[o3d.geometry.PointCloud], folder_path, method):
    matching_matrix = get_fragment_matchings(fragments, folder_path)

    data = []
    # Iterate over matching pairs.
    for a, b in zip(*np.nonzero(matching_matrix)):
        a_rotated = copy.deepcopy(fragments[a]).rotate(ROTATION, center=(0, 0, 0))
        b_rotated = copy.deepcopy(fragments[b]).rotate(ROTATION, center=(0, 0, 0))

        args.keypoint_method = method
        keypoints_a, _ = get_keypoints(a, np.array(a_rotated.points), np.array(a_rotated.normals),
                                    method=method, folder_path=folder_path,
                                    npoints=NUM_KEYPOINTS, tag='rotated')
        keypoints_b, _ = get_keypoints(b, np.array(b_rotated.points), np.array(b_rotated.normals),
                                    method=method, folder_path=folder_path,
                                    npoints=NUM_KEYPOINTS, tag='rotated')
        if a < b:  # Only once per pair.
            data.append((a, b, *calculate_score(keypoints_a[:, :3], keypoints_b[:, :3])))

    return data


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

    # object name, fragment, method, distances
    data_repeatability = []
    for i, pcd in enumerate(fragment_pcds):
        for method in KEYPOINT_METHODS:
            distances = evaluate_repeatability(i, pcd, args, method, folder_path)
            if method == 'iss':
                missing = 2 * NUM_KEYPOINTS - distances.shape[0]
                distances = np.pad(distances, (0, missing), mode='constant', constant_values=np.nan)
            data_repeatability.append((object_name, i, method, *distances))

    # object name, method, fragment_1, fragment_2, distances
    surface_repeatability = []
    for method in KEYPOINT_METHODS:
        data = evaluate_surface_repeatability(fragment_pcds, folder_path, method)
        print(f"folder {folder_path} has {len(data)} pairs.")
        for entry in data:
            if method == 'iss':
                missing = 2 + 2 * NUM_KEYPOINTS - len(entry)
                entry += missing * (np.nan,)
            surface_repeatability.append((object_name, method, *entry))

    return data_repeatability, surface_repeatability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate repeatability and different fragment matching of keypoints."
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--descriptor_method", type=str, default='fpfh', choices=['fpfh', 'pillar', 'fpfh_pillar'])
    args = parser.parse_args()

    paths = [os.path.abspath(os.path.join(args.path, folder)) for folder in os.listdir(args.path)]
    fn = functools.partial(process_folder, args)
    num_cpus = min(len(paths), 4)
    # with Pool(num_cpus) as p:
    #     # Data is [NUM_PROCS, 2, NUM_FRAGS * NUM_METHODS]
    #     data = list(tqdm(p.imap(fn, paths), total=len(paths)))

    data = []
    for p in paths:
        data += [process_folder(args, p)]

    repeatability = pd.DataFrame(data=[entry for sublist, _ in data for entry in sublist],
                                 columns=['object_name', 'fragment_idx', 'method',
                                          *[f'k_{i}' for i in range(2 * NUM_KEYPOINTS)]])
    repeatability.to_csv('keypoint_repeatability.csv')
    print(repeatability.sort_values(by='method'))

    sufrace_repeatability = pd.DataFrame(data=[entry for _, sublist in data for entry in sublist],
                                         columns=['object_name', 'method', 'fragment_1', 'fragment_2',
                                                  *[f'k_{i}' for i in range(2 * NUM_KEYPOINTS)]])
    sufrace_repeatability.to_csv('keypoint_surface_repeatability.csv')
    print(sufrace_repeatability.sort_values(by='method'))
