import os
from copy import deepcopy

import numpy as np
import open3d.cpu.pybind.geometry
from scipy.spatial import KDTree
from scipy.spatial.distance import norm
from sklearn.decomposition import PCA

from tools.neighborhoords import k_ring_delaunay_adaptive
from tools.tools import polyfit3d
from tools.transformation import centering_centroid
import open3d as o3d

def get_hybrid_keypoints(vertices, normals, n_neighbors, n_keypoints=512, sharp_percentage=0.6, mixture=0.7):
    c, sd = compute_smoothness_sd(vertices, normals, n_neighbors)
    c = np.array(c)
    sd = np.array(sd)

    idx_sorted_c = np.argsort(c)
    idx_sorted_sd = np.argsort(np.abs(sd))

    # get the first and last ones as best keypoints
    n_kpts_c = int(n_keypoints * mixture)
    n_kpts_sd = n_keypoints - n_kpts_c
    n_sharp = int(n_kpts_c * sharp_percentage)
    n_plan = n_kpts_c - n_sharp

    # filter out the ones which are just directly on planes (the Xe-5 is arbitrary but seemed to work okay)
    start_planar = next(i for i, item in enumerate(c[idx_sorted_c]) if item > 2.5e-5)
    # check to not take c kpts double (just empirically stable for now, not optimal)
    if start_planar + n_plan > len(vertices) - n_sharp:
        start_planar = next(i for i, item in enumerate(c[idx_sorted_c]) if item > 9e-6)

    planar_idx = idx_sorted_c[start_planar:n_plan + start_planar]
    sharp_idx = idx_sorted_c[-n_sharp:]

    sd_idx = idx_sorted_sd[-n_kpts_sd:]
    indices_to_keep = np.append(planar_idx, sharp_idx, axis=0)
    indices_to_keep = np.append(indices_to_keep, sd_idx, axis=0)

    kpts_planar = np.array(vertices[planar_idx])
    kpts_sharp = np.array(vertices[sharp_idx])
    kpts_sd = np.array(vertices[sd_idx])

    kpts = np.append(kpts_planar, kpts_sharp, axis=0)
    kpts = np.append(kpts, kpts_sd, axis=0)
    if len(kpts) != n_keypoints:
        exit("Failed calculating the keypoints!")
    scores_planar = np.array(c[planar_idx])
    scores_sharp = np.array(c[sharp_idx])
    scores = np.append(scores_planar, scores_sharp, axis=0)
    scores = scores / np.max(scores)

    scores_sd = sd[sd_idx]
    scores_sd = scores_sd / np.max(scores_sd)

    scores = np.append(scores, scores_sd, axis=0)
    scores = scores / np.max(scores)

    return np.column_stack((kpts, scores)), indices_to_keep


def compute_smoothness_sd(vertices, normals, n_neighbors):
    n_p = len(vertices)
    tree = KDTree(vertices)
    _, neighbors = tree.query(vertices, p=2, k=n_neighbors)

    c = []
    sd = []
    for hood in neighbors:
        # smoothness
        point = hood[0]  # closest point is always point itself
        neigh = hood[1:]
        diff = [[vertices[point] - vertices[n]] for n in neigh]
        c.append(norm(np.sum(diff, axis=0), 2) / (n_p * norm(vertices[point], 2)))
        # sd calculation
        n_p_i = normals[hood[0]]
        p_i_bar = np.mean(vertices[hood], axis=0)
        v = point - p_i_bar
        sd.append(np.dot(v, n_p_i))

    return c, sd


def compute_smoothness(vertices, n_neighbors):
    n_p = len(vertices)
    tree = KDTree(vertices)
    _, neighbors = tree.query(vertices, p=2, k=n_neighbors)

    c = []
    for hood in neighbors:
        point = hood[0]
        neigh = hood[1:]
        diff = [[vertices[point] - vertices[n]] for n in neigh]
        c.append(norm(np.sum(diff, axis=0), 2) / (n_p * norm(vertices[point], 2)))
    return c


def compute_SD_point(neighbourhood, points, normals, p_idx):
    p_i = points[p_idx]
    n_p_i = normals[p_idx]
    p_i_bar = np.mean(points[neighbourhood], axis=0)
    v = p_i - p_i_bar
    SD = np.dot(v, n_p_i)
    return SD


def get_SD_keypoints(vertices, normals, r=0.05, nkeypoints=256):
    """
    Returns the SD keypoints with a score value normalized.
    Following https://ieeexplore.ieee.org/document/9279208.
    """
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

    # subsample for big pointclouds
    if len(points) > 5000:
        samp_idx = np.random.choice(len(points), 5000, replace=False)
        points = points[samp_idx]

    # initialisation of the solution
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
        points_2D = points[:, :2] - points[i, :2]

        # fit a quadratic surface
        m = polyfit3d(points_2D[:, 0], points_2D[:, 1], points[:, 2], order=2)
        m = m.reshape((3, 3))

        # Compute the derivative
        fx2 = m[1, 0] * m[1, 0] + 2 * m[2, 0] * m[2, 0] + 2 * m[1, 1] * m[1, 1]  # A
        fy2 = m[1, 0] * m[1, 0] + 2 * m[1, 1] * m[1, 1] + 2 * m[0, 2] * m[0, 2]  # B
        fxfy = m[1, 0] * m[0, 1] + 2 * m[2, 0] * m[1, 1] + 2 * m[1, 1] * m[0, 2]  # C

        # Compute response
        resp[i] = fx2 * fy2 - fxfy * fxfy - k * (fx2 + fy2) * (fx2 + fy2)

    # Select interest points at local maxima
    candidate = []
    for i in neighborhood.keys():
        if resp[i] >= np.max(resp[neighborhood[i]]):
            candidate.append([i, resp[i]])
    # sort by decreasing order
    candidate.sort(reverse=True, key=lambda x: x[1])
    candidate = np.array(candidate)

    keypoint_indexes = np.array(
        candidate[:npoints, 0], dtype=np.int)

    return keypoint_indexes


def get_pillar_keypoints(vertices, n_neighbors, n_keypoints=512, sharp_percentage=0.5):
    c = np.array(compute_smoothness(vertices, n_neighbors))
    # normalize it and argsort to get lowest and highest values
    idx_sorted = np.argsort(c)
    # get the first and last ones as best keypoints
    n_sharp = int(n_keypoints * sharp_percentage)
    n_plan = n_keypoints - n_sharp

    # get start for planar regions
    start_planar = next((i for i, val in enumerate(c[idx_sorted]) if val > 5e-3), 0)
    planar_idx = idx_sorted[start_planar:n_plan + start_planar]
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


def get_iss_keypoints(vertices):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    return np.array(keypoints.points)


def get_keypoints(i, vertices, normals, desc_normal, desc_inv, args, folder_path, npoints, keypoints_only=False, tag=''):
    method = args.keypoint_method
    if len(tag) > 0:
        tag += '_'
    processed_path = os.path.join(args.path, folder_path, 'processed')
    keypoint_path = os.path.join(processed_path, 'keypoints', f'keypoints_{tag}{method}.{i}.npy')

    os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)

    if keypoints_only and os.path.exists(keypoint_path):
        keypoints = np.load(keypoint_path)
        return keypoints

    if not "pillar" in args.descriptor_method:
        kpts_desc_path_normal = os.path.join(processed_path, 'keypoint_descriptors',
                                             f'keypoint_descriptors_{tag}{method}_{args.descriptor_method}.{i}.npy')
        kpts_desc_path_inverted = os.path.join(processed_path, 'keypoint_descriptors_inverted',
                                               f'keypoint_descriptors_{tag}{method}_{args.descriptor_method}.{i}.npy')
    else:
        kpts_desc_path_normal = os.path.join(processed_path, 'pillar_keypoint_descriptors',
                                             f'keypoint_descriptors_{tag}{method}_{args.descriptor_method}.{i}.npy')
        kpts_desc_path_inverted = os.path.join(processed_path, 'pillar_keypoint_descriptors_inverted',
                                               f'keypoint_descriptors_{tag}{method}_{args.descriptor_method}.{i}.npy')
    os.makedirs(os.path.dirname(kpts_desc_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(kpts_desc_path_inverted), exist_ok=True)
    if os.path.exists(kpts_desc_path_normal) and os.path.exists(kpts_desc_path_inverted) and os.path.exists(keypoint_path):
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
    elif args.keypoint_method == 'iss':
        assert keypoints_only
        keypoints = get_iss_keypoints(vertices)
        keypoint_idxs = None
    else:
        raise NotImplementedError

    np.save(keypoint_path, keypoints)
    if not keypoints_only:
        kpt_desc_normal = desc_normal[keypoint_idxs]
        np.save(kpts_desc_path_normal, kpt_desc_normal)
        if args.keypoint_method == 'SD' or (args.keypoint_method == 'hybrid' and args.descriptor_method != "pillar"):
            kpt_desc_invert = desc_inv[keypoint_idxs]
            np.save(kpts_desc_path_inverted, kpt_desc_invert)

    return keypoints
