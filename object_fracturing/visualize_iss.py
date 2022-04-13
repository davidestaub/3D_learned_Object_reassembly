import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from tomlkit import key
from tools import *
from compas.geometry import Point, Pointcloud, closest_point_in_cloud, Line
import numpy as np
from compas_view2.app import App
import open3d as o3d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# ==============================================================================
# File
# ==============================================================================

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))

data = os.path.join(here,"training_data_example","connected_1vN")

frag1 = os.path.join(data,"fragments_1")
frag2 = os.path.join(data,"fragments_2")




# initialize viewer
#viewer = App()

explode = False

files_in = [ob for ob in os.listdir(frag1) if ob.endswith(".npy")]
files_in_obj = [ob for ob in os.listdir(frag1) if ob.endswith(".obj")]
file_in_nums = len(files_in)
print("There are", file_in_nums, "of input fragments.")





print("id  name")
for idx, val in enumerate(files_in):
    print(idx," ", val)

idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file = files_in[idx]
obj = files_in_obj[idx]

path_to_file = os.path.join(frag1,file)

mesh = Mesh.from_obj(os.path.join(frag1,obj))


# find the projected points
# convert the mesh to a pointcloud
vertices = np.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
mesh_points = []
pcd_points =  []

for vert in vertices:
    pcd_points.append([vert[0], vert[1], vert[2]])
    mesh_points.append(Point(x=vert[0],y=vert[1],z=vert[2]))
mesh_ptc = Pointcloud(mesh_points)

#closest_points = []
#for points in cloud:
    #closest_points.append(closest_point_in_cloud(points,mesh_ptc)[1])

#closest_cloud = Pointcloud(closest_points)
#viewer.add(closest_cloud, color=i_to_rgb(0.5, True))

# add lines:
#for idx in range(len(closest_cloud)):
    #line = Line(cloud[idx], closest_points[idx])
    #viewer.add(line)

# generate iss keypoints
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius=0.005, non_max_radius=0.005)

iss_points = []
for point in keypoints.points:
    iss_points.append(Point(x=point[0], y=point[1],z=point[2]))

iss_points = Pointcloud(iss_points)
#viewer.add(iss_points, color = i_to_rgb(0.1,True))


o3d.visualization.draw_geometries([pcd.points])



#viewer.run()