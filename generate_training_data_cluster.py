from asyncio import subprocess
from tools.transformation import centering_centroid
from tools.neighborhoords import k_ring_delaunay_adaptive
from tools.tools import polyfit3d, mesh_faces_to_triangles
import argparse
from copy import deepcopy
import os
from glob import glob
from typing import List
from sklearn.decomposition import PCA
import sys
import pyshot
import subprocess
from multiprocessing import Pool
from functools import partial
import shutil
import numpy as np
from compas.datastructures import Mesh
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import save_npz, csr_matrix
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


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
            matches = np.sum(cdist(fragments[i][:, :3], fragments[j][:, :3]) < 1e-3)

            # If there are more than 100 matches, the parts are considered neighbours.
            if matches > 300:
                matching_matrix[i, j] = matching_matrix[j, i] = 1

    np.save(matching_matrix_path, matching_matrix)
    return matching_matrix


def compute_SD_point(neighbourhood, points, normals, p_idx):
    p_i = points[p_idx]
    n_p_i = normals[p_idx]
    p_i_bar = np.mean(points[neighbourhood], axis=0)
    v = p_i - p_i_bar
    SD = np.dot(v, n_p_i)
    return SD


def get_SD_keypoints(vertices, normals, r=0.09, nkeypoints=256):
    """ returns the SD keypoints with a score value normalized"""
    n_points = len(vertices)
    tree = KDTree(vertices)
    # Compute SD
    SD = np.zeros((n_points))
    neighbourhoods = tree.query_ball_point(vertices, r, workers=-1)

    for i in range(n_points):
        neighbourhood = np.asarray(neighbourhoods[i])
        SD[i] = compute_SD_point(neighbourhood, vertices, normals, i)

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


def get_descriptors(i, vertices, faces, args, folder_path):
    # shesh
    method = args.descriptor_method
    mount_path = '/mnt/c/Users/mathi/OneDrive/Studium/ETH/MA2/3D Vision/Project/Old Project/3D_learned_Object_reassembly'
    primitive = folder_path.split('/')[-1]
    filename = ''.join([primitive, f'_cleaned.{i}.pcd'])

    if method == 'shot':
        descriptors = pyshot.get_descriptors(vertices, faces,
                                                radius=args.radius,
                                                local_rf_radius=args.local_rf_radius,
                                                min_neighbors=args.min_neighbors,
                                                n_bins=args.n_bins,
                                                double_volumes_sectors=args.double_volumes_sectors,
                                                use_interpolation=args.use_interpolation,
                                                use_normalization=args.use_normalization,
                                                )      
        descriptor_path = os.path.join(folder_path, 'processed', 'descriptors_all_points', f'descriptors_all_points_{method}.{i}.npy')              
        np.save(descriptor_path, descriptors)
        return descriptors

    if method == 'fpfh':
        method = args.descriptor_method
        in_path = os.path.join(mount_path, folder_path, 'cleaned', filename)
        out_path_prefix = os.path.join(mount_path, folder_path, 'processed', 'descriptors_all_points', f'descriptors_all_points_{method}.{i}')
        out_path_normal = '_'.join([out_path_prefix, 'normal.txt'])
        out_path_inverted = '_'.join([out_path_prefix, 'inverted.txt'])
        os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)

        fpfh_args = ['/home/mathi/PCL/Release/fpfh', in_path, out_path_normal, "False"]
        subprocess.check_call(fpfh_args)
        fpfh_args = ['/home/mathi/PCL/Release/fpfh', in_path, out_path_inverted, "True"]
        subprocess.check_call(fpfh_args)
        descriptors = [np.loadtxt(out_path_normal), np.loadtxt(out_path_inverted)]
        return descriptors



def get_keypoints(i, vertices, normals, descriptors, args, folder_path):
    method = args.keypoint_method
    desc_normal, desc_inv = descriptors
    
    keypoint_path = os.path.join(folder_path, 'processed', 'keypoints',f'keypoints_{method}.{i}.npy')
    kpts_desc_path_normal = os.path.join(folder_path, 'processed', 'keypoint_descriptors',f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
    kpts_desc_path_inverted = os.path.join(folder_path, 'processed', 'keypoint_descriptors_inverted',f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')

    os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(kpts_desc_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(kpts_desc_path_inverted), exist_ok=True)

    if os.path.exists(kpts_desc_path_normal) and os.path.exists(kpts_desc_path_inverted):
        keypoints = np.load(keypoint_path)
        return keypoints

    if args.keypoint_method == 'SD':
        keypoints, keypoint_idxs = get_SD_keypoints(vertices, normals, r=0.05)
        kpt_desc_normal = desc_normal[keypoint_idxs]
        kpt_desc_invert = desc_inv[keypoint_idxs]

        np.save(keypoint_path, keypoints)
        np.save(kpts_desc_path_normal, kpt_desc_normal)
        np.save(kpts_desc_path_inverted, kpt_desc_invert)
        return keypoints
    else:
        raise NotImplementedError


def process_folder(folder_path, args):
    object_name = os.path.basename(folder_path)
    shutil.rmtree(os.path.join(folder_path, 'processed'), ignore_errors=True)
    os.makedirs(os.path.join(folder_path, 'processed'), exist_ok=True)

    # TODO: reading from binary might be much faster according to Martin.
    obj_files = glob(os.path.join(folder_path, 'cleaned', '*.obj'))
    fragments_vertices = []
    fragments_faces = []
    fragments_normals = []

    # Load obj files in order, so fragments_*[0] has shard 0.
    num_fragments = len(obj_files)
    if num_fragments == 0:
        print(f"No fragments found in {folder_path}")
        return

    for i in range(num_fragments):
        mesh = Mesh.from_obj(os.path.join(folder_path, 'cleaned', f'{object_name}_cleaned.{i}.obj'))
        # Some faces are still polygons other than triangles :(
        mesh_faces_to_triangles(mesh)
        vertices, faces = mesh.to_vertices_and_faces()
        normals = [mesh.vertex_normal(vkey) for vkey in mesh.vertices()]

        # The only faces left are degenerate, i e same vertex is part of it twice. Filter them out.
        faces = list(filter(lambda vertex_list: len(vertex_list) == 3, faces))
        fragments_vertices.append(np.array(vertices))
        fragments_faces.append(np.array(faces))
        fragments_normals.append(np.array(normals))

    keypoints = []
    for i in range(num_fragments):
        fragment_descriptors = get_descriptors(i, fragments_vertices[i], fragments_faces[i], args, folder_path)
        frag_kpts = get_keypoints(i, fragments_vertices[i], fragments_normals[i], fragment_descriptors, args, folder_path)
        keypoints.append(frag_kpts)

    matching_matrix = get_fragment_matchings(fragments_vertices, folder_path)

    for i in range(num_fragments):
        for j in range(i):
            if matching_matrix[i, j]:
                keypoint_assignment = get_keypoint_assignment( keypoints[i][:,:3], keypoints[j][:,:3]).astype(int)
                print(f"Found {np.sum(keypoint_assignment)} matches!")
                # save the matching matrix as sparse scipy file
                name = f'match_matrix_{args.keypoint_method}_{args.descriptor_method}_{i}_{j}'
                path = os.path.join(folder_path, 'processed', 'matching', name)
                save_npz(path, csr_matrix(keypoint_assignment))

    # delete unecessary files again
    shutil.rmtree(os.path.join(folder_path, 'processed', 'descriptors_all_points'))

    del keypoints, matching_matrix, fragment_descriptors
    gc.collect()
    print(f'Processed folder {folder_path}')


def main():
    parser = argparse.ArgumentParser("generate_iss_keypoints_and_shot_descriptors")

    parser.add_argument("--keypoint_method", type=str,default='SD', choices=['iss', 'SD', 'harris'])
    parser.add_argument("--descriptor_method", type=str,default='fpfh', choices = ['shot', 'fpfh'])
    parser.add_argument("--data_dir", type=str, default='')
    # Args for SHOT descriptors.
    parser.add_argument("--radius", type=float, default=0.1),
    parser.add_argument("--local_rf_radius", default=0.1, type=float)
    parser.add_argument("--min_neighbors", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--double_volumes_sectors", action='store_true')
    parser.add_argument("--use_interpolation", action='store_true')
    parser.add_argument("--use_normalization", action='store_true')
    parser.add_argument("--invert_normals", default="True")
    args = parser.parse_args()

    args.local_rf_radius = args.radius if args.local_rf_radius is None else args.local_rf_radius
    args.data_dir = os.path.join(os.path.curdir, 'object_fracturing', 'data') if not args.data_dir else args.data_dir

    print(f'Data dir: {args.data_dir}')
    object_folders = glob(os.path.join(args.data_dir, '*'))
    
    with Pool(4) as p:
        p.map(partial(process_folder, args=args), object_folders)

if __name__ == '__main__':
    main()
