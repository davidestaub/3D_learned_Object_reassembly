import argparse
import os
from glob import glob
from typing import List

import numpy as np
import open3d as o3d
import pyshot
from compas.datastructures import Mesh, mesh_split_face
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import save_npz, csr_matrix


# from tools import *

# # create pointcloud from the mesh vertices
# vertices = np.array([part_mesh.vertex_coordinates(vkey) for vkey in part_mesh.vertices()])
# mesh_points = []
# pcd_points = []
#
# # get the points of the fragment as pure data and points for
# for vert in vertices:
#     pcd_points.append([vert[0], vert[1], vert[2]])
#     mesh_points.append(Point(x=vert[0], y=vert[1], z=vert[2]))
# part_ptc = Pointcloud(mesh_points)
#
# # generate iss points
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pcd_points)
# keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
#
# # make keypoints back to pointcloud in compas
# iss_ptc = []
# for point in keypoints.points:
#     iss_ptc.append(Point(x=point[0], y=point[1], z=point[2]))
# iss_ptc = Pointcloud(iss_ptc)
#
# # load the counter part
# counter_part = np.load(os.path.join(frag_2, file_counterpart))
#
# # generate a pointcloud
# counterpart_ptc = Pointcloud([Point(i[0], i[1], i[2]) for i in counter_part])
#
# # find closest points
# closest_points = []
# for points in iss_ptc:
#     closest_points.append(closest_point_in_cloud(points, counterpart_ptc))
#
# min_dist = min([item[0] for item in closest_points])
#
# closest_points_thresh = []
# for point in closest_points:
#     if point[0] < 10 * min_dist:
#         closest_points_thresh.append(point[1])
#
# closest_ptc = Pointcloud(closest_points_thresh)


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


# Copied from compas and modified to also take care of 5 sidede meshes.
def mesh_faces_to_triangles(mesh):
    """Convert all quadrilateral faces of a mesh to triangles by adding a diagonal edge.
    mesh : :class:`~compas.datastructures.Mesh` A mesh data structure.
    The mesh is modified in place.
    """

    def cut_off_traingle(fkey):
        attr = mesh.face_attributes(fkey)
        attr.custom_only = True
        vertices = mesh.face_vertices(fkey)
        # We skip degenerate faces because compas can't even handle deleting them.
        if len(vertices) >= 4 and len(vertices) == len(set(vertices)):
            a = vertices[0]
            c = vertices[2]
            t1, t2 = mesh_split_face(mesh, fkey, a, c)  # Cut off abc triangle.
            mesh.face_attributes(t1, attr.keys(), attr.values())
            mesh.face_attributes(t2, attr.keys(), attr.values())
            if fkey in mesh.facedata:
                del mesh.facedata[fkey]
            cut_off_traingle(t2)  # t2 still can have more than 3 vertices.

    for fkey in list(mesh.faces()):
        cut_off_traingle(fkey)


def get_iss_keypoints(vertices):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    keypoints = np.asarray(o3d.geometry.keypoint.compute_iss_keypoints(pcd).points)
    keypoint_idxs = np.where(cdist(vertices, keypoints, metric='cityblock') == 0)[0]
    return keypoints, keypoint_idxs

def compute_SD_point(neighbourhood, points, normals, p_idx):
		p_i = points[p_idx]
		n_p_i = normals[p_idx]
		p_i_bar = np.mean(points[neighbourhood], axis=0)
		v = p_i - p_i_bar
		SD = np.dot(v, n_p_i)
		return SD

def get_SD_keypoints(vertices, normals, r, kpts_fraction=0.1):
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
    if args.descriptor_method == 'shot':
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


def get_keypoints(i, vertices,normals, descriptors, args, folder_path):
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

    if args.keypoint_method == 'iss':
        keypoints, keypoint_idxs = get_iss_keypoints(vertices)
        keypoint_descriptors = descriptors[keypoint_idxs]
        np.save(keypoint_path, keypoints)
        np.save(keypoint_descriptors_path, keypoint_descriptors)
        return keypoints, keypoint_descriptors
    if args.keypoint_method == 'SD':
        keypoints, keypoint_idxs = get_SD_keypoints(vertices, normals, r=0.01)
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


def main():
    parser = argparse.ArgumentParser("generate_iss_keypoints_and_shot_descriptors")

    parser.add_argument("--keypoint_method", type=str, default='iss', choices=['iss', 'SD'])
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
    for f in object_folders:
        if os.path.isdir(f):
            process_folder(f, args)


if __name__ == '__main__':
    main()
