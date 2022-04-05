import os
from tools import *
from compas.geometry import Pointcloud
import numpy as np
from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
train_folder = os.path.join(here,'training_data','train','connected_1vN')
frag_1 = os.path.join(train_folder, 'fragments_1\\')
frag_2 = os.path.join(train_folder, 'fragments_2\\')
print(frag_1)
n_files = len(os.listdir(frag_1))

# generate random number to watch the pair
random_num = np.random.randint(n_files)
random_num = str(random_num)

# load the pointclouds
pc_1 = np.load(frag_1+random_num+".npy")
pc_2 = np.load(frag_2+random_num+".npy")


# initialize viewer
viewer = App()

mesh_1 = Pointcloud(pc_1[:,:3])
mesh_2 = Pointcloud(pc_2[:,:3])

viewer.add(mesh_1, color=[0,0,0])
viewer.add(mesh_2, color=[255,0,255])


viewer.run()

