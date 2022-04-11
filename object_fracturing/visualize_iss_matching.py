import os
from compas.datastructures import Mesh, mesh_transform_numpy
from tools import *
from compas.geometry import Point, Pointcloud, closest_point_in_cloud, distance_point_point
import numpy as np
from compas_view2.app import App
import open3d as o3d
import compas.geometry as cg

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
data = os.path.join(here, 'training_data_example', 'connected_1vN')
frag_1 = os.path.join(data, 'fragments_1')
frag_2 = os.path.join(data, 'fragments_2')


# initialize viewer
viewer = App()

find_closest_points = False
match_iss_points = True
centering = True 

files_in_npy = [ob for ob in os.listdir(frag_1) if ob.endswith(".npy")]
files_in_obj = [ob for ob in os.listdir(frag_1) if ob.endswith(".obj")]

print("id  name")
for idx, val in enumerate(files_in_npy):
    print(idx, " ", val)

idx = int(input(
    "Enter the index of the subfolder in data where the shards are located:\n"))
file_npy = files_in_npy[idx]
file_obj = files_in_obj[idx]

# generate the counterpart name
file_counterpart = file_npy.split(
    'part')[0] + "counterpart" + file_npy.split('part')[1].split('.')[0] + ".npy"

# load the fragment part
part_mesh = Mesh.from_obj(os.path.join(frag_1, file_obj))

# create pointcloud from the mesh vertices
vertices = np.array([part_mesh.vertex_coordinates(vkey)
                    for vkey in part_mesh.vertices()])
mesh_points = []
pcd_points = []

# get the points of the fragment as pure data and points for
for vert in vertices:
    pcd_points.append([vert[0], vert[1], vert[2]])
    mesh_points.append(Point(x=vert[0], y=vert[1], z=vert[2]))
part_ptc = Pointcloud(mesh_points)

# generate iss points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

# make keypoints back to pointcloud in compas
iss_ptc = []
for point in keypoints.points:
    iss_ptc.append(Point(x=point[0], y=point[1], z=point[2]))
iss_ptc = Pointcloud(iss_ptc)

# load the counter part
counter_part = np.load(os.path.join(frag_2, file_counterpart))

# generate a pointcloud
counterpart_ptc = []
counter_part_points = []
for point in counter_part:
    counterpart_ptc.append(Point(point[0], point[1], point[2]))
    counter_part_points.append([point[0], point[1], point[2]])
counterpart_ptc = Pointcloud(counterpart_ptc)

# find iss points on counter parts
pcd_counterpart = o3d.geometry.PointCloud()
pcd_counterpart.points = o3d.utility.Vector3dVector(counter_part_points)
keypoints_counterpart = o3d.geometry.keypoint.compute_iss_keypoints(
    pcd_counterpart)
counter_part_iss_ptc = Pointcloud(
    [Point(p[0], p[1], p[2]) for p in keypoints_counterpart.points])

# find closest points for iss keypoints of the part in the whole counterparts
if find_closest_points:
    # find closest points
    closest_points = []
    for points in iss_ptc:
        closest_points.append(closest_point_in_cloud(points, counterpart_ptc))
    # find the minimal distance between a iss point and closest point
    # of counterpart
    min_dist = min([item[0] for item in closest_points])
    # filter the closest poitns to be more meaningful
    # cause some are very far away
    closest_points_thresh = []
    for point in closest_points:
        if point[0] < 7 * min_dist:
            closest_points_thresh.append(point[1])

    closest_ptc = Pointcloud(closest_points_thresh)
    viewer.add(closest_ptc, color=[50, 50, 0])

# find close points in iss points within a threshold
close_iss_matchings = []
if match_iss_points:
    for part_point in iss_ptc:
        for counterpart_point in counter_part_iss_ptc:
            dist = distance_point_point(part_point, counterpart_point)
            if dist < 1e-2:
                close_iss_matchings.append(counterpart_point)
                close_iss_matchings.append(part_point)
close_iss_matchings = Pointcloud(close_iss_matchings)  
    

# centering the points
if centering:
    # center mesh
    center = part_mesh.centroid()
    vec =  cg.Vector(-center[0], -center[1], -center[2])
    T = cg.Translation.from_vector(vec)
    mesh_transform_numpy(part_mesh, T)
    # center pointclouds
    close_iss_matchings.transform(T)
    counter_part_iss_ptc.transform(T)
    iss_ptc.transform(T)


# add objects to viewer
if match_iss_points:
    viewer.add(close_iss_matchings, color=[255,0,0])
else:
    viewer.add(counter_part_iss_ptc, color=[50, 0, 100])

viewer.add(part_mesh, facecolor=[0, 23, 80])
viewer.add(iss_ptc, color=[0, 255, 0])

viewer.run()
