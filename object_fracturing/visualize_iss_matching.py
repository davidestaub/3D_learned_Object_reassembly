from ntpath import join
import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from tools import *
from compas.geometry import Point, Pointcloud, closest_point_in_cloud, Line
import numpy as np
from compas_view2.app import App
import open3d as o3d

# ==============================================================================
# File
# ==============================================================================
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
data = os.path.join(here,'training_data_example','connected_1vN')
frag_1 = os.path.join(data,'fragments_1')
frag_2 = os.path.join(data,'fragments_2')


# initialize viewer
viewer = App()

explode = False

files_in_npy = [ob for ob in os.listdir(frag_1) if ob.endswith(".npy")]
files_in_obj = [ob for ob in os.listdir(frag_1) if ob.endswith(".obj")]

print("id  name")
for idx, val in enumerate(files_in_npy):
    print(idx," ", val)

idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file_npy = files_in_npy[idx]
file_obj = files_in_obj[idx]

# generate the counterpart name
file_counterpart = file_npy.split('part')[0] + "counterpart" + file_npy.split('part')[1].split('.')[0] + ".npy"

# load the fragment part
part_mesh = Mesh.from_obj(os.path.join(frag_1,file_obj))

# create pointcloud from the mesh vertices
vertices = np.array([part_mesh.vertex_coordinates(vkey) for vkey in part_mesh.vertices()])
mesh_points = []
pcd_points =  []

# get the points of the fragment as pure data and points for 
for vert in vertices:
    pcd_points.append([vert[0], vert[1], vert[2]])
    mesh_points.append(Point(x=vert[0],y=vert[1],z=vert[2]))
part_ptc = Pointcloud(mesh_points)

# generate iss points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

# make keypoints back to pointcloud in compas
iss_ptc = []
for point in keypoints.points:
    iss_ptc.append(Point(x=point[0],y=point[1],z=point[2]))
iss_ptc = Pointcloud(iss_ptc)

# load the counter part
counter_part = np.load(os.path.join(frag_2,file_counterpart))

# generate a pointcloud
counterpart_ptc = Pointcloud([Point(i[0], i[1], i[2]) for i in counter_part])

# find closest points
closest_points = []
for points in iss_ptc:
    closest_points.append(closest_point_in_cloud(points,counterpart_ptc))

min_dist = min([item[0] for item in closest_points])

closest_points_thresh = []
for point in closest_points:
    if point[0] < 10 * min_dist:
        closest_points_thresh.append(point[1])

closest_ptc = Pointcloud(closest_points_thresh)

# add objects to viewer
viewer.add(part_mesh, facecolor=i_to_rgb(0.1, True))
viewer.add(iss_ptc, color=i_to_rgb(0.3, True))
viewer.add(closest_ptc, color=i_to_rgb(0.7, True))


viewer.run()