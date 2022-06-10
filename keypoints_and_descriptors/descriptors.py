import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.distance import norm


def calculate_fpfh(vertices, normals):
    pcd_normal = o3d.geometry.PointCloud()
    pcd_normal.normals = o3d.utility.Vector3dVector(normals)
    pcd_normal.points = o3d.utility.Vector3dVector(vertices)
    pcd_invert = o3d.geometry.PointCloud()
    pcd_invert.normals = o3d.utility.Vector3dVector(-1 * normals)
    pcd_invert.points = o3d.utility.Vector3dVector(vertices)

    if o3d.__version__ == "0.9.0.0":
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


def fpfh_pillar_encoder(pcd, normals, sample_size=10):
    pcd = np.array(pcd)
    tree = KDTree(pcd)
    dist, neighbourhoods = tree.query(pcd, sample_size)
    # calculate fpfh for the whole cloud
    fpfh_norm, fpfh_inv = calculate_fpfh(pcd, normals)

    pillars_norm = []
    pillars_inv = []
    for idx, hood in enumerate(neighbourhoods):
        cog = np.mean(pcd[hood], axis=0)
        feat_norm = fpfh_norm[hood]
        feat_inv = fpfh_inv[hood]
        # adding 3 more features to get to 36
        # add distances to keypoint
        dist_kpt = dist[idx]
        feat_norm = np.column_stack((feat_norm, dist_kpt))
        feat_inv = np.column_stack((feat_inv, dist_kpt))
        # add distance to cog
        dist_cog = [norm(p - cog) for p in pcd[hood]]
        feat_norm = np.column_stack((feat_norm, dist_cog))
        feat_inv = np.column_stack((feat_inv, dist_cog))
        # pad rest with zeros
        pad = np.zeros((sample_size, 1))
        feat_norm = np.column_stack((feat_norm, pad))
        feat_inv = np.column_stack((feat_inv, pad))

        pillars_norm.append(feat_norm)
        pillars_inv.append(feat_inv)

    return np.array(pillars_norm), np.array(pillars_inv)


def pillar_encoder(kpts, pcd, sample_size=10):
    pcd = np.array(pcd)
    tree = KDTree(pcd)
    dist, neighbourhoods = tree.query(kpts, sample_size)
    pillars = []
    for hood in neighbourhoods:
        pillars.append(pcd[hood])

    # calculate features
    features = []
    for i, pillar in enumerate(pillars):
        feature = []
        pillar_center = np.mean(pillar, axis=0)
        pillar_kpt = pillar[0]
        for j, point in enumerate(pillar):
            feature.append(np.concatenate((point, point - pillar_center, [norm(point)], point - pillar_kpt)))
        features.append(feature)

    return np.array(features)


def get_descriptors(vertices, normals, method: str):
    if method == 'fpfh':
        desc_norm, desc_inv = calculate_fpfh(vertices, normals)
        return desc_norm, desc_inv

    if method == "pillar":
        desc = pillar_encoder(vertices, vertices)
        return desc, None

    if method == "fpfh_pillar":
        pillar_norm, pillar_inv = fpfh_pillar_encoder(vertices, normals)
        return pillar_norm, pillar_inv


def save_descriptors(descriptors, descriptors_inv, object_folder_path, keypoint_method, descriptor_method, fragment_id, tag=''):
    if len(tag) > 0:
        tag += '_'
    processed_path = os.path.join(object_folder_path, 'processed')

    kpts_desc_path_normal = os.path.join(processed_path, 'keypoint_descriptors',
                                         f'keypoint_descriptors_{tag}{keypoint_method}_{descriptor_method}.{fragment_id}.npy')
    os.makedirs(os.path.dirname(kpts_desc_path_normal), exist_ok=True)
    np.save(kpts_desc_path_normal, descriptors)
    if descriptors_inv:
        kpts_desc_path_inverted = os.path.join(processed_path, 'keypoint_descriptors_inverted',
                                               f'keypoint_descriptors_{tag}{keypoint_method}_{descriptor_method}.{fragment_id}.npy')
        os.makedirs(os.path.dirname(kpts_desc_path_inverted), exist_ok=True)
        np.save(kpts_desc_path_inverted, descriptors_inv)