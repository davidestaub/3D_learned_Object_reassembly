import argparse
from copy import deepcopy
import os
from glob import glob
from typing import List
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, cpu_count
from tools.tools import dot_product, length, polyfit3d, mesh_faces_to_triangles
from tools.neighborhoords import k_ring_delaunay_adaptive
from tools.transformation import centering_centroid
import shutil
import numpy as np
import pyshot
from compas.datastructures import Mesh
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import save_npz, csr_matrix

def get_fragment_matchings(fragments: List[np.array], folder_path: str):
    object_name = os.path.basename(folder_path)
    match_path = os.path.join(folder_path,'processed', 'matching')
    os.makedirs(match_path, exist_ok=True)

    matching_matrix_path = os.path.join(match_path, f'{object_name}_matching_matrix.npy')

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
            if matches > 100:
                # print(f'{name}: {i} and {j} match!')
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

def get_SD_keypoints(vertices, normals, r=0.1, kpts_fraction=0.1):
    nkeypoints = int(len(vertices) * kpts_fraction)
    n_points = len(vertices)
    tree = KDTree(vertices)
    # Compute SD
    SD = np.zeros((n_points))
    neighbourhoods = tree.query_ball_point(vertices, r, workers=-1)

    for i in range(n_points):
        neighbourhood = np.asarray(neighbourhoods[i])
        SD[i] = compute_SD_point(neighbourhood, vertices, normals, i)

    indices_to_keep = np.argsort(np.abs(SD))[-nkeypoints:]
    keypoints = vertices[indices_to_keep]
    return keypoints, indices_to_keep

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
        
        #best fitting point
        points_pca = PCA(n_components=3).fit_transform(np.transpose(points_centred))
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
        fx2  = m[1, 0]*m[1, 0] + 2*m[2, 0]*m[2, 0] + 2*m[1, 1]*m[1, 1]  # A
        fy2  = m[1, 0]*m[1, 0] + 2*m[1, 1]*m[1, 1] + 2*m[0, 2]*m[0, 2]  # B
        fxfy = m[1, 0]*m[0, 1] + 2*m[2, 0]*m[1, 1] + 2*m[1, 1]*m[0, 2]  # C

        # Compute response
        resp[i] = fx2*fy2 - fxfy*fxfy - k*(fx2 + fy2)*(fx2 + fy2)

    #Select interest points at local maxima
    candidate = []
    for i in neighborhood.keys() :
        if resp[i] >= np.max(resp[neighborhood[i]]) :
            candidate.append([i, resp[i]])
    #sort by decreasing order
    candidate.sort(reverse=True, key=lambda x:x[1])
    candidate = np.array(candidate)
    
    #Method 1 : fraction
    keypoint_indexes = np.array(candidate[:int(fraction*len(points)), 0], dtype=np.int)
    labels_fraction[keypoint_indexes] = 1

    return keypoint_indexes

def get_keypoint_assignment(keypoints1, keypoints2, threshold=0.05):
    dists = cdist(keypoints1, keypoints2)
    close_enough_mask = np.min(dists, axis=0) < threshold
    closest = np.argmin(dists, axis=0)

    keypoint_assignment = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))
    keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1

    return keypoint_assignment


def get_descriptors(i, vertices, faces, args, folder_path):
    method = args.descriptor_method

    descriptor_path = os.path.join(folder_path, 'processed', 'descriptors_all_points',
                                   f'descriptors_all_points_{method}.{i}.npy')
    os.makedirs(os.path.dirname(descriptor_path), exist_ok=True)

    if os.path.exists(descriptor_path):
        descriptors = np.load(descriptor_path)
        return descriptors
    if method == 'shot':
        print("get_descriptor")
        descriptors = pyshot.get_descriptors(vertices, faces,
                                             radius=args.radius,
                                             local_rf_radius=args.local_rf_radius,
                                             min_neighbors=args.min_neighbors,
                                             n_bins=args.n_bins,
                                             double_volumes_sectors=args.double_volumes_sectors,
                                             use_interpolation=args.use_interpolation,
                                             use_normalization=args.use_normalization,
                                             )                       
        np.save(descriptor_path, descriptors)
        return descriptors
    else:
        raise NotImplementedError


def get_keypoints(i, vertices, normals, descriptors, args, folder_path):
    method = args.keypoint_method

    keypoint_path = os.path.join(folder_path, 'processed', 'keypoints',
                                 f'keypoints_{method}.{i}.npy')
    keypoint_descriptors_path = os.path.join(folder_path, 'processed', 'keypoint_descriptors',
                                             f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
    os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(keypoint_descriptors_path), exist_ok=True)
    if os.path.exists(keypoint_path) and os.path.exists(keypoint_descriptors_path):
        keypoints = np.load(keypoint_path)
        keypoint_descriptors = np.load(keypoint_descriptors_path)
        return keypoints, keypoint_descriptors

    if args.keypoint_method == 'SD':
        keypoints, keypoint_idxs = get_SD_keypoints(vertices, normals, r=0.01)
        keypoint_descriptors = descriptors[keypoint_idxs]
        np.save(keypoint_path, keypoints)
        np.save(keypoint_descriptors_path, keypoint_descriptors)
        return keypoints, keypoint_descriptors
    if args.keypoint_method == 'harris':
        keypoint_idxs = get_harris_keypoints(vertices)
        keypoints = vertices[keypoint_idxs]
        keypoint_descriptors = descriptors[keypoint_idxs]
        np.save(keypoint_path, keypoints)
        np.save(keypoint_descriptors_path, keypoint_descriptors)
        return keypoints, keypoint_descriptors   
    else:
        raise NotImplementedError


def process_folder(folder_path, args):
    object_name = os.path.basename(folder_path)
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
    descriptors = []
    for i in range(num_fragments):
        fragment_descriptors = get_descriptors(i, fragments_vertices[i], fragments_faces[i], args, folder_path)
        fragment_keypoints, fragment_keypoint_descriptors = get_keypoints(i, fragments_vertices[i], fragments_normals[i],
                                                                          fragment_descriptors, args, folder_path)
        keypoints.append(fragment_keypoints)
        descriptors.append(fragment_keypoint_descriptors)

    matching_matrix = get_fragment_matchings(fragments_vertices, folder_path)

    for i in range(num_fragments):
        for j in range(i):
            if matching_matrix[i, j]:
                keypoint_assignment = get_keypoint_assignment(keypoints[i], keypoints[j]).astype(int)
                print(f"{keypoint_assignment.sum()} matching keypoint in pair {i} {j}")
                # save the matching matrix as sparse scipy file
                name = f'match_matrix_{args.keypoint_method}_{args.descriptor_method}_{i}_{j}'
                path = os.path.join(folder_path,'processed', 'matching', name)
                save_npz(path, csr_matrix(keypoint_assignment))
    
    # delete unecessary files again
    shutil.rmtree(os.path.join(folder_path, 'processed', 'descriptors_all_points'))


def main():
    parser = argparse.ArgumentParser("generate_iss_keypoints_and_shot_descriptors")

    parser.add_argument("--keypoint_method", type=str, default='SD', choices=['iss', 'SD', 'harris'])
    parser.add_argument("--descriptor_method", type=str, default='shot', choices=['shot'])

    parser.add_argument("--data_dir", type=str, default='')

    # Args for SHOT descriptors.
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--local_rf_radius", type=float, default=None)
    parser.add_argument("--min_neighbors", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--double_volumes_sectors", action='store_true')
    parser.add_argument("--use_interpolation", action='store_true')
    parser.add_argument("--use_normalization", action='store_true')
    args = parser.parse_args()

    args.local_rf_radius = args.radius if args.local_rf_radius is None else args.local_rf_radius
    args.data_dir = os.path.join(os.path.curdir, 'object_fracturing', 'data') if not args.data_dir else args.data_dir

    object_folders = glob(os.path.join(args.data_dir, '*'))
    Parallel(n_jobs=cpu_count(only_physical_cores=True))(delayed(process_folder)(f, args) for f in object_folders if os.path.isdir(f))

if __name__ == '__main__':
    main()
