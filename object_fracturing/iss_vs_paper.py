import os
from compas.datastructures import Mesh, mesh_transform_numpy
from tools.tools import *
from compas.geometry import Point, Pointcloud, closest_point_in_cloud, distance_point_point
import numpy as np
from compas_view2.app import App
import open3d as o3d
import compas.geometry as cg

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
data = os.path.join(here, 'data', 'bottle_10_seed_1')
iss_path = os.path.join(data, 'iss_keypoints')
paper_path = os.path.join(data, 'paper_keypoints')
mesh_path = os.path.join(data,'subdv')

iss_files = os.listdir(iss_path)
paper_files = os.listdir(paper_path)
mesh_files = os.listdir(mesh_path)


idx = 0

iss_npy = np.load(os.path.join(iss_path, iss_files[idx]))
paper_npy = np.load(os.path.join(paper_path, paper_files[idx]))
iss_kpts = iss_npy[:,:3]
paper_kpts = paper_npy[:,:3]



iss_ptc = Pointcloud(iss_kpts)
paper_ptc = Pointcloud(paper_kpts)

viewer = App()

if True:
    mesh = Mesh().from_obj(os.path.join(mesh_path, mesh_files[1 +2*idx]))
    # center mesh
    center = mesh.centroid()
    vec =  cg.Vector(-center[0], -center[1], -center[2])
    T = cg.Translation.from_vector(vec)
    mesh_transform_numpy(mesh, T)
    # center pointclouds
    iss_ptc.transform(T)
    paper_ptc.transform(T)

print(mesh.face_plane(0))

viewer.add(iss_ptc,color=[100,0,0]) # red
viewer.add(mesh,facecolor=[0,100,0]) # blue
viewer.add(paper_ptc,color=[0,100,100]) # yellow

viewer.show()
