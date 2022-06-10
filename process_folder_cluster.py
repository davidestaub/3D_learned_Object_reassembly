import argparse
import os
import shutil
import time
from glob import glob

import numpy as np
import open3d as o3d
from scipy.sparse import save_npz, csr_matrix

from keypoints_and_descriptors.descriptors import get_descriptors, save_descriptors
from keypoints_and_descriptors.keypoints import get_keypoints
from keypoints_and_descriptors.utils import get_fragment_matchings, get_keypoint_assignment


def process_folder(folder_path, args):
    start_time = time.time()
    object_name = os.path.basename(folder_path)
    # shutil.rmtree(os.path.join(args.path, folder_path,'processed'), ignore_errors=True)
    # os.makedirs(os.path.join(args.path, folder_path,'processed'), exist_ok=True)

    obj_files = glob(os.path.join(args.path, folder_path, 'cleaned', '*.pcd'))
    frag_vert = []
    frag_norm = []

    num_fragments = len(obj_files)
    if num_fragments == 0:
        print(f"No fragments found in {folder_path}")
        return
    fragment_pcds = []
    for i in range(num_fragments):
        file_path = os.path.join(args.path, folder_path, 'cleaned', f'{object_name}_cleaned.{i}.pcd')
        fragment_pcds.append(o3d.io.read_point_cloud(file_path))

    # extract vertices for gt mathing
    for pcd in fragment_pcds:
        vertices = np.array(pcd.points)
        frag_vert.append(vertices)

    matching_matrix = get_fragment_matchings(frag_vert, os.path.join(args.path, folder_path))

    # clean up matrices
    frag_vert = []
    frag_norm = []
    pcd_centers = []
    # center the pointclouds and reextract
    for pcd in fragment_pcds:
        c = pcd.get_center()
        pcd_centers.append(c)
        pcd = pcd.translate(-1 * c)
        vertices = np.array(pcd.points)
        normals = np.array(pcd.normals)
        frag_vert.append(vertices)
        frag_norm.append(normals)

    keypoints = []
    full_object_folder_path = os.path.join(args.path.args, folder_path)
    for i in range(num_fragments):
        desc_n, desc_inv = get_descriptors(frag_vert[i], frag_norm[i], args.descriptor_method)
        frag_kpts, keypoints_idxs = get_keypoints(i, frag_vert[i], frag_norm[i], args.keypoint_method,
                                                  folder_path=full_object_folder_path, npoints=512)
        kpts_desc_n = desc_n[keypoints_idxs]
        if desc_inv:
            kpts_desc_inv = desc_inv[keypoints_idxs]
        else:
            kpts_desc_inv = None
        save_descriptors(kpts_desc_n, kpts_desc_inv, full_object_folder_path, args.keypoint_method,
                         args.decriptor_method, fragment_id=i)
        keypoints.append(frag_kpts)

    # log for matches
    log = []

    # create the groundtruth
    for i in range(num_fragments):
        for j in range(i):
            if matching_matrix[i, j]:
                name = f'match_matrix_{args.keypoint_method}_{args.descriptor_method}_{i}_{j}'
                path = os.path.join(args.path, folder_path, 'processed', 'matching', name)
                if os.path.exists(path):
                    continue
                kpts_i = keypoints[i][:, :3]
                kpts_j = keypoints[j][:, :3]
                # translate back
                kpts_i = kpts_i + pcd_centers[i]
                kpts_j = kpts_j + pcd_centers[j]
                keypoint_assignment = get_keypoint_assignment(kpts_i, kpts_j).astype(int)
                log.append(f"Found {np.sum(keypoint_assignment)} matches for {i}-{j}!")
                # save the matching matrix as sparse scipy file
                save_npz(path, csr_matrix(keypoint_assignment))

    logpath = os.path.join(folder_path, 'log.txt')
    if log:
        with open(logpath, 'w') as f:
            f.write('\n'.join(log))

    # delete unecessary files again
    try:
        shutil.rmtree(os.path.join(args.path, folder_path, 'processed', 'descriptors_all_points'))
    except:
        pass

    print(f'Processed folder {folder_path} in {time.time() - start_time}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn jobs for blender_auto_fracture_cluster.py")
    parser.add_argument("--path", type=str)
    parser.add_argument("--keypoint_method", type=str, default='hybrid', choices=['SD', 'sticky', 'hybrid'])
    parser.add_argument("--descriptor_method", type=str, default='fpfh', choices=['fpfh', 'pillar', 'fpfh_pillar'])
    args = parser.parse_args()
    process_folder(os.path.abspath(args.path), args)
