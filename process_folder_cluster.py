from email.policy import default
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

def fpfh_pillar_encoder(pcd, normals, sample_size = 10):
    pcd = np.array(pcd)
    tree = KDTree(pcd)
    dist, neighbourhoods = tree.query(pcd, sample_size)
    # calculate fpfh for the whole cloud
    fpfh_norm, fpfh_inv = calculate_fpfh(pcd, normals)

    pillars_norm = []
    pillars_inv  = []
    for idx, hood in enumerate(neighbourhoods):
        cog = np.mean(pcd[hood], axis = 0)
        feat_norm = fpfh_norm[hood]
        feat_inv  = fpfh_inv[hood]
        # adding 3 more features to get to 36
        # add distances to keypoint
        dist_kpt = dist[idx]
        feat_norm = np.column_stack((feat_norm, dist_kpt))
        feat_inv = np.column_stack((feat_inv, dist_kpt))
        # add distance to cog
        dist_cog = [norm(p-cog) for p in pcd[hood]]
        feat_norm = np.column_stack((feat_norm, dist_cog))
        feat_inv = np.column_stack((feat_inv, dist_cog))
        # pad rest with zeros
        pad = np.zeros((sample_size, 1))
        feat_norm = np.column_stack((feat_norm, pad))
        feat_inv = np.column_stack((feat_inv, pad))

        pillars_norm.append(feat_norm)
        pillars_inv.append(feat_inv)

    return np.array(pillars_norm), np.array(pillars_inv)

def pillar_encoder(kpts, pcd, sample_size = 10):
    pcd = np.array(pcd)
    # downsample to x y coordinates
    #kpts_2d = kpts[:,:2]
    #pcd_2d = pcd[:,:2]
    tree = KDTree(pcd)
    dist, neighbourhoods = tree.query(kpts, sample_size)
    pillars = []
    for hood in neighbourhoods:
        pillars.append(pcd[hood])
    
    # calculate features
    features = []
    for pillar in pillars:
        feature = []
        pillar_center = np.mean(pillar, axis=0)
        pillar_kpt = pillar[0]
        for point in pillar:
            feature.append(np.concatenate((point,point-pillar_center,[norm(point)],point-pillar_kpt)))
        features.append(feature)

    return np.array(features)

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
            if matches > 600:
                print(f"Matched fragment {i} and {j}!")
                matching_matrix[i, j] = matching_matrix[j, i] = 1

    np.save(matching_matrix_path, matching_matrix)
    return matching_matrix

def get_hybrid_keypoints(vertices, normals, n_neighbors, n_keypoints = 512, sharp_percentage = 0.6, mixture = 0.7):
    c, sd = compute_smoothness_sd(vertices, normals, n_neighbors)
    c = np.array(c)
    sd = np.array(sd)

    idx_sorted_c = np.argsort(c)
    idx_sorted_sd = np.argsort(np.abs(sd))

    # get the first and last ones as best keypoints
    n_kpts_c = int(n_keypoints*mixture)
    n_kpts_sd = n_keypoints - n_kpts_c
    n_sharp = int(n_kpts_c*sharp_percentage)
    n_plan = n_kpts_c - n_sharp

    # filter out the ones which are just directly on planes (the Xe-5 is arbitrary but seemed to work okay)
    start_planar = next(i for i, item in enumerate(c[idx_sorted_c]) if item > 2.5e-5)
    # check to not take c kpts double (just empirically stable for now, not optimal)
    if start_planar+n_plan > len(vertices) - n_sharp:
        start_planar = next(i for i, item in enumerate(c[idx_sorted_c]) if item > 9e-6)

    planar_idx = idx_sorted_c[start_planar:n_plan+start_planar]
    sharp_idx = idx_sorted_c[-n_sharp:]

    sd_idx = idx_sorted_sd[-n_kpts_sd:]
    indices_to_keep = np.append(planar_idx, sharp_idx, axis = 0)
    indices_to_keep = np.append(indices_to_keep, sd_idx, axis = 0)
    
    kpts_planar = np.array(vertices[planar_idx])
    kpts_sharp = np.array(vertices[sharp_idx])
    kpts_sd = np.array(vertices[sd_idx])

    kpts = np.append(kpts_planar, kpts_sharp, axis = 0)
    kpts = np.append(kpts, kpts_sd, axis = 0)
    if len(kpts) != n_keypoints:
        exit("Failed calculating the keypoints!")
    scores_planar = np.array(c[planar_idx])
    scores_sharp = np.array(c[sharp_idx])
    scores = np.append(scores_planar, scores_sharp, axis=0)
    scores = scores / np.max(scores)

    scores_sd = sd[sd_idx]
    scores_sd = scores_sd / np.max(scores_sd)

    scores = np.append(scores, scores_sd, axis = 0)
    scores = scores / np.max(scores)
    
    return np.column_stack((kpts, scores)), indices_to_keep


def compute_smoothness_sd(vertices, normals, n_neighbors):
    n_p = len(vertices)
    tree =  KDTree(vertices)
    _, neighbors = tree.query(vertices, p=2, k=n_neighbors)

    c = []
    sd = []
    for hood in neighbors:
        #smoothness
        point = hood[0] # closest point is always point itself
        neigh = hood[1:]
        diff = [[vertices[point]-vertices[n]] for n in neigh]
        c.append(norm(np.sum(diff, axis=0),2) / (n_p*norm(vertices[point], 2)))
        # sd calculation
        n_p_i = normals[hood[0]]
        p_i_bar = np.mean(vertices[hood], axis=0)
        v = point - p_i_bar
        sd.append(np.dot(v, n_p_i))

    return c, sd

def get_pillar_keypoints(vertices, n_neighbors, n_keypoints=512, sharp_percentage = 0.5):
    c = np.array(compute_smoothness(vertices, n_neighbors))
    # normalize it and argsort to get lowest and highest values
    idx_sorted = np.argsort(c)
    # get the first and last ones as best keypoints
    n_sharp = int(n_keypoints*sharp_percentage)
    n_plan = n_keypoints - n_sharp

    # get start for planar regions
    start_planar = next((i for i, val in enumerate(c[idx_sorted]) if val > 5e-3), 0)
    planar_idx = idx_sorted[start_planar:n_plan+start_planar]
    sharp_idx = idx_sorted[-n_sharp:]
    indices_to_keep = np.append(planar_idx, sharp_idx)

    kpts_planar = np.array(vertices[planar_idx])
    kpts_sharp = np.array(vertices[sharp_idx])
    kpts = np.append(kpts_planar, kpts_sharp, axis=0)

    scores_planar = np.array(c[planar_idx])
    scores_sharp = np.array(c[sharp_idx])
    scores = np.append(scores_planar, scores_sharp, axis=0)
    scores = scores / np.max(scores)
    return np.column_stack((kpts, scores)), indices_to_keep

def calculate_fpfh(vertices, normals):
        pcd_normal = o3d.geometry.PointCloud()
        pcd_normal.normals = o3d.utility.Vector3dVector(normals)
        pcd_normal.points = o3d.utility.Vector3dVector(vertices)
        pcd_invert = o3d.geometry.PointCloud()
        pcd_invert.normals = o3d.utility.Vector3dVector(-1 * normals)
        pcd_invert.points = o3d.utility.Vector3dVector(vertices)

        if o3dV == "0.9.0.0":
            pcd_fpfh_norm = o3d.registration.compute_fpfh_feature(
                pcd_normal,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=8))
            pcd_fpfh_inv = o3d.registration.compute_fpfh_feature(
                pcd_invert,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=8))
        else:
            pcd_fpfh_norm = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_normal,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=8))
            pcd_fpfh_inv = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_invert,
                o3d.geometry.KDTreeSearchParamHybrid(radius=25, max_nn=8))
        desc_norm = np.array(pcd_fpfh_norm.data).T
        desc_inv = np.array(pcd_fpfh_inv.data).T
        return desc_norm, desc_inv

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


def get_SD_keypoints(vertices, normals, r=0.05, nkeypoints=256):
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


def get_harris_keypoints(vertices, npoints=512):
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
    # keypoint_indexes = np.array(
    #     candidate[:int(fraction*len(points)), 0], dtype=np.int)
    # labels_fraction[keypoint_indexes] = 1
    keypoint_indexes = np.array(
         candidate[:npoints, 0], dtype=np.int)

    return keypoint_indexes


def get_keypoint_assignment(keypoints1, keypoints2, threshold=3e-2):
    dists = cdist(keypoints1, keypoints2)
    close_enough_mask = np.min(dists, axis=0) < threshold
    closest = np.argmin(dists, axis=0)

    keypoint_assignment = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))
    keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1

    return keypoint_assignment


def get_descriptors(vertices, normals, args):

    method = args.descriptor_method

    if method == 'fpfh':
        desc_norm, desc_inv = calculate_fpfh(vertices, normals)
        return desc_norm, desc_inv
    
    if method == "pillar":
        desc = pillar_encoder(vertices, vertices)
        return desc, None

    if method == "fpfh_pillar":
        pillar_norm, pillar_inv = fpfh_pillar_encoder(vertices, normals)
        return pillar_norm, pillar_inv


def get_keypoints(i, vertices, normals, desc_normal, desc_inv, args, folder_path, npoints, save=True):
    method = args.keypoint_method
    processed_path = os.path.join(args.path, folder_path, 'processed')
    keypoint_path = os.path.join(processed_path, 'keypoints', f'keypoints_{method}.{i}.npy')
    if not "pillar" in args.descriptor_method:
        kpts_desc_path_normal = os.path.join(processed_path, 'keypoint_descriptors',f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
        kpts_desc_path_inverted = os.path.join(processed_path,'keypoint_descriptors_inverted', f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
    else:
        kpts_desc_path_normal = os.path.join(processed_path, 'pillar_keypoint_descriptors',f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')
        kpts_desc_path_inverted = os.path.join(processed_path,'pillar_keypoint_descriptors_inverted', f'keypoint_descriptors_{method}_{args.descriptor_method}.{i}.npy')

    if save:
        os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(kpts_desc_path_normal), exist_ok=True)
        os.makedirs(os.path.dirname(kpts_desc_path_inverted), exist_ok=True)

    if os.path.exists(kpts_desc_path_normal) and os.path.exists(kpts_desc_path_inverted):
        keypoints = np.load(keypoint_path)
        return keypoints

    if args.keypoint_method == 'SD':
        keypoints, keypoint_idxs = get_SD_keypoints(vertices, normals, r=0.1, nkeypoints=npoints)
    elif args.keypoint_method == 'sticky':
        keypoints, keypoint_idxs = get_pillar_keypoints(vertices, 12, npoints)
    elif args.keypoint_method == 'hybrid':
        keypoints, keypoint_idxs = get_hybrid_keypoints(vertices, normals, 12, npoints)
    elif args.keypoint_method == 'harris':
        keypoint_idxs = get_harris_keypoints(vertices, npoints)
        keypoints = vertices[keypoint_idxs]
    else:
        raise NotImplementedError

    if save:
        np.save(keypoint_path, keypoints)
        kpt_desc_normal = desc_normal[keypoint_idxs]
        np.save(kpts_desc_path_normal, kpt_desc_normal)
        if args.keypoint_method == 'SD' or (args.keypoint_method == 'hybrid' and args.descriptor_method != "pillar"):
            kpt_desc_invert = desc_inv[keypoint_idxs]
            np.save(kpts_desc_path_inverted, kpt_desc_invert)

    return keypoints


def process_folder(folder_path, args):
    start_time = time.time()
    object_name = os.path.basename(folder_path)
    #shutil.rmtree(os.path.join(args.path, folder_path,'processed'), ignore_errors=True)
    #os.makedirs(os.path.join(args.path, folder_path,'processed'), exist_ok=True)

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
        pcd = pcd.translate(-1*c)
        vertices = np.array(pcd.points)
        normals = np.array(pcd.normals)
        frag_vert.append(vertices)
        frag_norm.append(normals)
    
    keypoints = []
    for i in range(num_fragments):
        desc_n, desc_inv = get_descriptors(frag_vert[i], frag_norm[i], args)
        frag_kpts = get_keypoints(i, frag_vert[i], frag_norm[i], desc_n, desc_inv, args, folder_path, 512)
        keypoints.append(frag_kpts)

    # log for matches
    log = []

    # create the groundtruth
    for i in range(num_fragments):
        for j in range(i):
            if matching_matrix[i, j]:
                name = f'match_matrix_{args.keypoint_method}_{args.descriptor_method}_{i}_{j}'
                path = os.path.join(args.path, folder_path,'processed', 'matching', name)
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

    logpath = os.path.join(folder_path,'log.txt')
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
    parser.add_argument("--keypoint_method", type=str,default='hybrid', choices=['SD', 'sticky', 'hybrid'])
    parser.add_argument("--descriptor_method", type=str,default='fpfh', choices=['fpfh', 'pillar', 'fpfh_pillar'])
    args = parser.parse_args()
    process_folder(os.path.abspath(args.path), args)
