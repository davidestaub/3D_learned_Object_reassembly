import argparse
import copy
import functools
import math
import os
from glob import glob
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import open3d as o3d
import pandas as pd
from open3d.cpu.pybind.geometry import PointCloud
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from process_folder_cluster import get_keypoints, get_fragment_matchings

np.random.seed(42)

# A random rotation.
angles = np.random.uniform(low=0, high=2 * math.pi, size=(3,))
ROTATION = PointCloud().get_rotation_matrix_from_xyz(angles)
REVERSE_ROTATION = PointCloud().get_rotation_matrix_from_xyz(-angles)
KEYPOINT_METHODS = ['SD', 'sticky', 'hybrid', 'iss']  # , 'harris']
NUM_KEYPOINTS = 512


def calculate_score(keypoints1, keypoints2):
    tree = KDTree(keypoints1)
    dists_to_closest1, _ = tree.query(keypoints2)
    tree = KDTree(keypoints2)
    dists_to_closest2, _ = tree.query(keypoints1)
    closest = np.concatenate([dists_to_closest1, dists_to_closest2])
    return closest


def evaluate_repeatability(fragment_idx, pcd, keypoints, args, method, folder_path):
    c = pcd.get_center()
    pcd = pcd.translate(-1 * c)
    pcd_rotated = copy.deepcopy(pcd).rotate(ROTATION, center=(0, 0, 0))

    args.keypoint_method = method
    keypoints_rotated = get_keypoints(fragment_idx, np.array(pcd_rotated.points), np.array(pcd_rotated.normals),
                                      args=args, desc_normal=None, desc_inv=None, folder_path=folder_path,
                                      npoints=NUM_KEYPOINTS,
                                      keypoints_only=True, tag='rotated')

    kpts_pcd = PointCloud(o3d.utility.Vector3dVector(keypoints[:, :3]))
    kpts_pcd_rotated = PointCloud(o3d.utility.Vector3dVector(keypoints_rotated[:, :3]))
    kpts_pcd_rotated = kpts_pcd_rotated.rotate(REVERSE_ROTATION)

    closest = calculate_score(kpts_pcd.points, kpts_pcd_rotated.points)

    return closest


def evaluate_surface_repeatability(fragments: List[o3d.geometry.PointCloud], keypoints, folder_path, method):
    matching_matrix = get_fragment_matchings(fragments, folder_path)

    data = []
    # Iterate over matching pairs.
    for a, b in zip(*np.nonzero(matching_matrix)):
        a_rotated = copy.deepcopy(fragments[a]).rotate(ROTATION, center=(0, 0, 0))
        b_rotated = copy.deepcopy(fragments[b]).rotate(ROTATION, center=(0, 0, 0))

        args.keypoint_method = method
        keypoints_a = get_keypoints(a, np.array(a_rotated.points), np.array(a_rotated.normals),
                                    args=args, desc_normal=None, desc_inv=None, folder_path=folder_path,
                                    npoints=NUM_KEYPOINTS,
                                    keypoints_only=True, tag='rotated')
        keypoints_b = get_keypoints(b, np.array(b_rotated.points), np.array(b_rotated.normals),
                                    args=args, desc_normal=None, desc_inv=None, folder_path=folder_path,
                                    npoints=NUM_KEYPOINTS,
                                    keypoints_only=True, tag='rotated')
        if a < b:  # Only once per pair.
            data.append((a, b, *calculate_score(keypoints_a, keypoints_b)))

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

    all_keypoints: Dict[str, List[PointCloud]] = {}  # method: {idx: kpts}
    for method in KEYPOINT_METHODS:
        all_keypoints[method] = []
        for i, f in enumerate(fragment_pcds):
            args.keypoint_method = method
            keypoints = get_keypoints(i, np.array(f.points), np.array(f.normals), args=args, desc_normal=None,
                                      desc_inv=None, folder_path=folder_path, npoints=NUM_KEYPOINTS,
                                      keypoints_only=True)
            all_keypoints[method].append(keypoints)

    # object name, fragment, method, distances
    data_repeatability = []
    for i, pcd in enumerate(fragment_pcds):
        for method in KEYPOINT_METHODS:
            distances = evaluate_repeatability(i, pcd, all_keypoints[method][i], args, method, folder_path)
            if method == 'iss':
                missing = 2 * NUM_KEYPOINTS - distances.shape[0]
                distances = np.pad(distances, (0, missing), mode='constant', constant_values=np.nan)
            data_repeatability.append((object_name, i, method, *distances))

    # object name, method, fragment_1, fragment_2, distances
    surface_repeatability = []
    for method in KEYPOINT_METHODS:
        data = evaluate_surface_repeatability(fragment_pcds, all_keypoints[method], folder_path, method)
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
