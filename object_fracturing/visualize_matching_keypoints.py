import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from tools import *
from compas.geometry import Point, Pointcloud, distance_point_point
import numpy as np
from compas_view2.app import App

# chose a data folder
here = os.path.dirname(os.path.abspath(__file__))
data_list = os.listdir(os.path.join(here, 'data'))
print("id  name")
for idx, val in enumerate(data_list):
    print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
subfolder = data_list[idx]
ROOT = os.path.join(here, 'data', subfolder)

# get the other folders
os.chdir(ROOT)
os.chdir('..')
DATAROOT = os.path.join(os.getcwd())
CLEANED = os.path.join(ROOT, 'cleaned')
KPTS_IN = os.path.join(ROOT, 'keypoints')

viewer = App()

# load all fragments
fragment_idx = 0
fragment_files = [file for file in  os.listdir(CLEANED) if file.endswith('.obj')]
fragment_num = len(fragment_files)
# add all fragments to the visualizer
for fragment in fragment_files:
    mesh = Mesh().from_obj(os.path.join(CLEANED, fragment))
    viewer.add(mesh, facecolor=i_to_rgb(fragment_idx/fragment_num, True))
    fragment_idx += 1

# load all the keypoints
kpt_files = [file for file in  os.listdir(KPTS_IN) if file.endswith('.npy')]
clouds = []
for pointcloud in kpt_files:
    kpt_in = np.load(os.path.join(KPTS_IN, pointcloud))
    kpt_in = kpt_in[:,:3]
    clouds.extend(kpt_in)

# go through all points and check distance
n_kpts = len(clouds)
close_points = []
for i in range(n_kpts):
    for j in range(i+1,n_kpts):
        dist = distance_point_point(clouds[i], clouds[j])
        if dist < 0.001:
            close_points.append(clouds[i])
            close_points.append(clouds[j])

keypoints_all = Pointcloud(close_points)
viewer.add(keypoints_all, color=[100,0,0])
viewer.run()
