import os
from secrets import choice
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from matplotlib.pyplot import show
from sympy import primitive
from tools import *
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg
from compas.geometry import Point, Pointcloud
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

# chose a fragments
data_list = os.listdir(CLEANED)
print("id  name")
for idx, val in enumerate(data_list):
    if val.endswith('.obj'):
        print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file = data_list[idx]
file_path = os.path.join(CLEANED, file)

# extract the corresponding filename
kpts_file = file.replace('cleaned.', 'kpts_')
kpts_file = kpts_file.replace('obj', 'npy')
kpts_file_path = os.path.join(KPTS_IN, kpts_file)

mesh = Mesh().from_obj(file_path)
kpts = np.load(kpts_file_path)
kpts = Pointcloud(kpts)

# initialize viewer
viewer = App()
viewer.add(mesh)
viewer.add(kpts, color = [100,0,0])

viewer.run()
