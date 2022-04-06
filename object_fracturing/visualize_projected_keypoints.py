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
ROOT = select_folder(mode='keypoints')
# get the data folder
os.chdir(ROOT)
os.chdir('..')
DATAROOT = os.path.join(os.getcwd())
KPTS_IN = os.path.join(ROOT, 'input\\')
KPTS_OUT = os.path.join(ROOT, 'output\\')



# initialize viewer
viewer = App()

explode = False

files_in = [ob for ob in os.listdir(KPTS_IN) if ob.endswith(".npy")]
file_in_nums = len(files_in)
print("There are", file_in_nums, "of input fragments.")

files_out = [ob for ob in os.listdir(KPTS_OUT) if ob.endswith(".npy")]
file_out_nums = len(files_out)
print("There are", file_in_nums, "of output fragments.")

if file_in_nums != file_out_nums:
    exit("There are not the same amount of inputs as outputs!")



print("id  name")
for idx, val in enumerate(files_out):
    print(idx," ", val)

idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file = files_out[idx]


# try loading a coresponding obj file (might change that later cause the naming has a npy in accidentaly)
obj_name = file.split('_kpts')[0]

try:
    mesh = Mesh.from_obj(os.path.join(KPTS_IN,obj_name))
except:
    print("Didn't find any obj files in the input folder!")
    if bool(input("Shall they be searched automatically?\n1:yes\n0:No\n")):
        object_primitive = obj_name.split('_subdv')[0]
        obj_name = obj_name.split('npy')[0] + "obj"
        automatic_in = os.path.join(DATAROOT,object_primitive,'subdv',obj_name)
        if not os.path.exists(automatic_in):
            exit("Error: There is no folder:", automatic_in)
        else:
            mesh = Mesh.from_obj(automatic_in)
            viewer.add(mesh)

# generate a pointcloud
kpts = np.load(os.path.join(KPTS_OUT,file))
points_compas = []
for item in kpts:
    points_compas.append(Point(item[0],item[1],item[2]))

cloud = Pointcloud(points_compas)
viewer.add(cloud, color=i_to_rgb(1, True))

# find the projected points
# convert the mesh to a pointcloud
vertices = np.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
mesh_points = []
pcd_points =  []

for vert in vertices:
    pcd_points.append([vert[0], vert[1], vert[2]])
    mesh_points.append(Point(x=vert[0],y=vert[1],z=vert[2]))
mesh_ptc = Pointcloud(mesh_points)

closest_points = []
for points in cloud:
    closest_points.append(closest_point_in_cloud(points,mesh_ptc)[1])

closest_cloud = Pointcloud(closest_points)
viewer.add(closest_cloud, color=i_to_rgb(0.5, True))

# add lines:
for idx in range(len(closest_cloud)):
    line = Line(cloud[idx], closest_points[idx])
    viewer.add(line)

# generate iss keypoints
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

iss_points = []
for point in keypoints.points:
    iss_points.append(Point(x=point[0], y=point[1],z=point[2]))

iss_points = Pointcloud(iss_points)
viewer.add(iss_points, color = i_to_rgb(0.1,True))

viewer.run()