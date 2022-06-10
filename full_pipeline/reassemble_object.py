"""
This script executes the whole pipeline:
    1. Keypoint extraction.
    2. Descriptor generation.
    3. Keypoint matching.
    4. Reassembly based on keypoint matchings.
"""
import argparse
import os
from glob import glob
import sys
import inspect

# setup paths for windows compability
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# setup default values for cube and weights
default_cube = os.path.join(currentdir,'example_data', 'cube_10_seed_0')
default_weights = os.path.join(parentdir, 'neural_network', 'weights', 'model_weights_best.pth')

import numpy as np
import open3d as o3d

from neural_network.predict import predict
from keypoints_and_descriptors.descriptors import get_descriptors, save_descriptors
from keypoints_and_descriptors.keypoints import get_keypoints
from object_reassembly.fractured_object import FracturedObject
from object_reassembly.reassembly_main import pairwise_reassembly, full_reassembly


def load_object(folder_path):
    object_name = os.path.basename(folder_path)
    pcd_files = glob(os.path.join(folder_path, 'cleaned', '*.pcd'))

    num_fragments = len(pcd_files)
    if num_fragments == 0:
        print(f"No fragments found in {folder_path}")
        return []

    fragment_pcds = []
    for i in range(num_fragments):
        # Load them in ascending order.
        file_path = os.path.join(folder_path, 'cleaned', f'{object_name}_cleaned.{i}.pcd')
        pcd = o3d.io.read_point_cloud(file_path)
        c = pcd.get_center()
        pcd = pcd.translate(-1 * c)
        fragment_pcds.append(pcd)

    return object_name, fragment_pcds





def main():
    parser = argparse.ArgumentParser(description="Execute the whole reassembly pipeline.")
    parser.add_argument("--object_dir", type=str, default= default_cube, help='Path to folder with object fragments in .obj format')
    parser.add_argument("--model", type=str, default=default_weights, help="Path to saved model")
    parser.add_argument('--pairwise', action='store_true', help='Perform and visualize only pairwise reassembly.')
    parser.add_argument('--ground_truth', action='store_true', help='Use ground truth matches to reassemble the object.')
    args = parser.parse_args()

    print(f'Loading object from {args.object_dir}.')
    name, fragments = load_object(args.object_dir)
    print(f'Loaded object {name} with {len(fragments)} fragments.')

    keypoint_method = 'hybrid'
    descriptor_method = 'pillar'
    print(f'Generating {keypoint_method} keypoints and {descriptor_method} descriptors...')

    for i, pcd in enumerate(fragments):
        vertices = np.array(pcd.points)
        normals = np.array(pcd.normals)
        descriptors, descriptors_inverted = get_descriptors(vertices, normals, descriptor_method)
        keypoints, keypoint_idxs = get_keypoints(i, vertices, normals, keypoint_method, args.object_dir, npoints=512)
        kpts_desc_n = descriptors[keypoint_idxs]
        if descriptors_inverted:
            kpts_desc_inv = descriptors_inverted[keypoint_idxs]
        else:
            kpts_desc_inv = None
        save_descriptors(kpts_desc_n, kpts_desc_inv, args.object_dir, keypoint_method, descriptor_method,
                         fragment_id=i)
    print(f'Done.')

    if not args.ground_truth:
        print(f'Predicting keypoint matches with GNN...')
        predict(default_weights, args.object_dir, single_object=True)
        print(f'Done.')

    print('Reassembling the object...')
    obj = FracturedObject(path=args.object_dir, graph_matching_method='mst')
    obj.load_object()
    obj.load_matches(use_ground_truth=args.ground_truth)
    if args.pairwise:
        pairwise_reassembly(obj)
    else:
        full_reassembly(obj)


if __name__ == '__main__':
    main()
