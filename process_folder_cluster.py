from sklearn import neighbors
from tools.transformation import centering_centroid
from tools.neighborhoords import k_ring_delaunay_adaptive
from tools.tools import polyfit3d
from copy import deepcopy
import os
import argparse
from glob import glob
from typing import List
from sklearn.decomposition import PCA
import shutil
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, norm
from scipy.sparse import save_npz, csr_matrix
import open3d as o3d
import time

o3dV = o3d.__version__

def get_fragment_matchings(fragments: List[np.array], folder_path: str):
    object_name = os.path.basename(folder_path)
    match_path = os.path.join(folder_path, 'processed', 'matching')
    os.makedirs(match_path, exist_ok=True)

    matching_matrix_path = os.path.join(
        match_path, f'{object_name}_matching_matrix.npy')

    # If matching is calculated already, use it.
    if os.path.exists(matching_matrix_path):
        matching_matrix = np.load(matching_matrix_path)
        return matching_matrix

    # Otherwise compute and save matchings.
    num_parts = len(fragments)
    matching_matrix = np.zeros((num_parts, num_parts))
    for i in range(num_parts):
        for j in range(i):
            # Search for corresponding points in two parts (distance below a treshold).
            matches = np.sum(
                cdist(fragments[i][:, :3], fragments[j][:, :3]) < 1e-3)

            # If there are more than 100 matches, the parts are considered neighbours.
            if matches > 300:
                matching_matrix[i, j] = matching_matrix[j, i] = 1

    np.save(matching_matrix_path, matching_matrix)
    return matching_matrix


def get_pillar_keypoints(vertices, n_neighbors, nkeypoints=512, sharp_percentage = 0.5):
    c = np.array(compute_smoothness(vertices, n_neighbors))
    idx_sorted = np.argsort(c)
    # get the first and last ones as best keypoints
    n_sharp = int(nkeypoints*sharp_percentage)
    n_plan = nkeypoints - n_sharp

    planar_idx = idx_sorted[-n_plan:]
    sharp_idx = idx_sorted[:n_sharp]
    indices_to_keep = np.append(planar_idx, sharp_idx)

    kpts_planar = np.array(vertices[planar_idx])
    kpts_sharp = np.array(vertices[sharp_idx])
    kpts = np.append(kpts_planar, kpts_sharp, axis=0)

    scores_planar = np.array(c[planar_idx])
    scores_sharp = np.array(c[sharp_idx])
    scores = np.append(scores_planar, scores_sharp, axis=0)
    return np.column_stack((kpts, scores)), indices_to_keep

def compute_smoothness(vertices, n_neighbors):
    n_p = len(vertices)
    tree =  KDTree(vertices)
    _, neighbors = tree.query(vertices,p=2, k=n_neighbors)

    c = []
    for hood in neighbors:
        point = hood[0]
        neigh = hood[1:]
        diff = [[vertices[point]-vertices[n]] for n in neigh]
        c.append(norm(np.sum(diff, axis=0),2) / (n_p*norm(vertices[point], 2)))
    return c

def compute_SD_point(neighbourhood, points, normals, p_idx):
    p_i = points[p_idx]
    n_p_i = normals[p_idx]
    p_i_bar = np.mean(points[neighbourhood], axis=0)
    v = p_i - p_i_bar
    SD = np.dot(v, n_p_i)
    return SD


def get_SD_keypoints(vertices, normals, r=0.1, nkeypoints=256):
    """ returns the SD keypoints with a score value normalized"""
    n_points = len(vertices)
    tree = KDTree(vertices)
    
    # Compute SD
    SD = np.zeros((n_points))
    neighbourhoods = tree.query_ball_point(vertices, r)

    for i in range(n_points):
        SD[i] = compute_SD_point(np.asarray(
            neighbourhoods[i]), vertices, normals, i)

    indices_to_keep = np.argsort(np.abs(SD))[-nkeypoints:]
    keypoints = np.array(vertices[indices_to_keep])
    scores = np.array(np.abs(SD)[indices_to_keep])
    scores = scores / np.max(scores)
    scores = scores[:, None]
    return np.append(keypoints, scores, axis=1), indices_to_keep


def get_harris_keypoints(vertices):
    points = deepcopy(vertices)
    # parameters
    delta = 0.025
    k = 0.04
    fraction = 0.1

    # subsample for big pointclouds
    if len(points) > 5000:
        samp_idx = np.random.choice(len(points), 5000, replace=False)
        points = points[samp_idx]

    # initialisation of the solution
    labels_fraction = np.zeros(len(points))
    resp = np.zeros(len(points))

    # compute neighborhood
    neighborhood = k_ring_delaunay_adaptive(points, delta)

    for i in neighborhood.keys():
        points_centred, _ = centering_centroid(points)

        # best fitting point
        points_pca = PCA(n_components=3).fit_transform(
            np.transpose(points_centred))
        _, eigenvectors = np.linalg.eigh(points_pca)

        # rotate the cloud
        for i in range(points.shape[0]):
            points[i, :] = np.dot(np.transpose(eigenvectors), points[i, :])
        # restrict to XY plane and translate

        points_2D = points[:, :2]-points[i, :2]

        # fit a quadratic surface
        m = polyfit3d(points_2D[:, 0], points_2D[:, 1], points[:, 2], order=2)
        m = m.reshape((3, 3))

        # Compute the derivative
        fx2 = m[1, 0]*m[1, 0] + 2*m[2, 0]*m[2, 0] + 2*m[1, 1]*m[1, 1]  # A
        fy2 = m[1, 0]*m[1, 0] + 2*m[1, 1]*m[1, 1] + 2*m[0, 2]*m[0, 2]  # B
        fxfy = m[1, 0]*m[0, 1] + 2*m[2, 0]*m[1, 1] + 2*m[1, 1]*m[0, 2]  # C

        # Compute response
        resp[i] = fx2*fy2 - fxfy*fxfy - k*(fx2 + fy2)*(fx2 + fy2)

    # Select interest points at local maxima
    candidate = []
    for i in neighborhood.keys():
        if resp[i] >= np.max(resp[neighborhood[i]]):
            candidate.append([i, resp[i]])
    # sort by decreasing order
    candidate.sort(reverse=True, key=lambda x: x[1])
    candidate = np.array(candidate)

    # Method 1 : fraction
    keypoint_indexes = np.array(
        candidate[:int(fraction*len(points)), 0], dtype=np.int)
    labels_fraction[keypoint_indexes] = 1

    return keypoint_indexes


def get_keypoint_assignment(keypoints1, keypoints2, threshold=0.001):
    dists = cdist(keypoints1, keypoints2)
    close_enough_mask = np.min(dists, axis=0) < threshold
    closest = np.argmin(dists, axis=0)

    keypoint_assignment = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))
    keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1

    return keypoint_assignment


def get_descriptors(i, vertices, normals, args, folder_path):
    # shesh
    method = args.descriptor_method
    primitive = folder_path.split('/')[-1]

    filename = ''.join([primitive, f'_cleaned.{i}.pcd'])
    in_path = os.path.join(folder_path, 'cleaned', filename)
    out_path_prefix = os.path.join(
        folder_path, 'processed', 'descriptors_all_points', f'descriptors_all_points_{method}.{i}')
    out_path_normal = '_'.join([out_path_prefix, 'normal.txt'])
    out_path_inverted = '_'.join([out_path_prefix, 'inverted.txt'])
    os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)

    if method == 'fpfh':
        pcd_normal = o3d.geometry.PointCloud()
        pcd_normal.normals = o3d.utility.Vector3dVector(normals)
        pcd_normal.points = o3d.utility.Vector3dVector(vertices)
        pcd_invert = o3d.geometry.PointCloud()
        pcd_invert.normals = o3d.utility.Vector3dVector(-1 * normals)
        pcd_invert.points = o3d.utility.Vector3dVector(vertices)

        if o3dV == "0.9.0":
            pcd_fpfh_norm = o3d.registration.compute_fpfh_feature(
                pcd_normal,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=16))
            pcd_fpfh_inv = o3d.registration.compute_fpfh_feature(
                pcd_invert,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=16))
        else:
            pcd_fpfh_norm = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_normal,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=16))
            pcd_fpfh_inv = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_invert,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=16)) 
        
        desc_norm = np.array(pcd_fpfh_norm.data).T
        desc_inv = np.array(pcd_fpfh_inv.data).T

        np.savetxt(out_path_normal, desc_norm)
        np.savetxt(out_path_inverted, desc_inv)
        return desc_norm, desc_inv


def get_keypoints(i, vertices, normals, desc_normal, desc_inv, args, folder_path):
    method = args.keypoint_method

    keypoint_path = os.path.join(
        args.path, folder_path, 'processed', 'keypoints', f'keypoints_{method}.{i}.npy')
    kpts_desc_path_normal = os.path.join(args.path, folder_path, 'processed', 'keypoint_descriptors',
                                         f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
    kpts_desc_path_inverted = os.path.join(args.path, folder_path, 'processed',
                                           'keypoint_descriptors_inverted', f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')

    os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(kpts_desc_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(kpts_desc_path_inverted), exist_ok=True)

    if os.path.exists(kpts_desc_path_normal) and os.path.exists(kpts_desc_path_inverted):
        keypoints = np.load(keypoint_path)
        return keypoints

    if args.keypoint_method == 'SD':
        keypoints, keypoint_idxs = get_SD_keypoints(vertices, normals, r=0.1)
        kpt_desc_normal = desc_normal[keypoint_idxs]
        kpt_desc_invert = desc_inv[keypoint_idxs]

        np.save(keypoint_path, keypoints)
        np.save(kpts_desc_path_normal, kpt_desc_normal)
        np.save(kpts_desc_path_inverted, kpt_desc_invert)
        return keypoints
    
    if args.keypoint_method == 'sticky':
        keypoints, keypoint_idxs = get_pillar_keypoints(vertices, 8)
        kpt_desc_normal = desc_normal[keypoint_idxs]
        kpt_desc_invert = desc_inv[keypoint_idxs]

        np.save(keypoint_path, keypoints)
        np.save(kpts_desc_path_normal, kpt_desc_normal)
        np.save(kpts_desc_path_inverted, kpt_desc_invert)
        return keypoints
        
    else:
        raise NotImplementedError


def process_folder(folder_path, args):
    start_time = time.time()
    object_name = os.path.basename(folder_path)
    #shutil.rmtree(os.path.join(args.path, folder_path,'processed'), ignore_errors=True)
    #os.makedirs(os.path.join(args.path, folder_path,'processed'), exist_ok=True)

    obj_files = glob(os.path.join(
        args.path, folder_path, 'cleaned', '*.pcd'))
    fragments_vertices = []
    fragments_normals = []

    num_fragments = len(obj_files)
    if num_fragments == 0:
        print(f"No fragments found in {folder_path}")
        return

    for i in range(num_fragments):
        file_path = os.path.join(
            args.path, folder_path, 'cleaned', f'{object_name}_cleaned.{i}.pcd')
        pcd = o3d.io.read_point_cloud(file_path)
        vertices = np.array(pcd.points)
        normals = np.array(pcd.normals)
        fragments_vertices.append(vertices)
        fragments_normals.append(normals)

    keypoints = []
    for i in range(num_fragments):
        desc_n, desc_inv = get_descriptors(
            i, fragments_vertices[i], fragments_normals[i], args, folder_path)
        frag_kpts = get_keypoints(
            i, fragments_vertices[i], fragments_normals[i], desc_n, desc_inv, args, folder_path)
        keypoints.append(frag_kpts)

    matching_matrix = get_fragment_matchings(
        fragments_vertices, os.path.join(args.path, folder_path))

    for i in range(num_fragments):
        for j in range(i):
            if matching_matrix[i, j]:
                keypoint_assignment = get_keypoint_assignment(
                    keypoints[i][:, :3], keypoints[j][:, :3]).astype(int)
                print(f"Found {np.sum(keypoint_assignment)} matches!")
                # save the matching matrix as sparse scipy file
                name = f'match_matrix_{args.keypoint_method}_{args.descriptor_method}_{i}_{j}'
                path = os.path.join(args.path, folder_path,
                                    'processed', 'matching', name)
                save_npz(path, csr_matrix(keypoint_assignment))

    # delete unecessary files again
    try:
        shutil.rmtree(os.path.join(args.path, folder_path,
                      'processed', 'descriptors_all_points'))
    except:
        pass

    print(f'Processed folder {folder_path} in {time.time() - start_time}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spawn jobs for blender_auto_fracture_cluster.py"
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--keypoint_method", type=str,default='SD', choices=['SD', 'sticky'])
    parser.add_argument("--descriptor_method", type=str,default='fpfh', choices=['shot', 'fpfh'])
    args = parser.parse_args()
    process_folder(os.path.abspath(args.path), args)
