import os
import numpy as np
import shutil
import sys
from joblib import Parallel, delayed
from compas.datastructures import Mesh, mesh_transform_numpy
from compas.datastructures import mesh_explode
from compas.datastructures import mesh_subdivide_corner
import compas.geometry as cg
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
dataroot = os.path.join(here, 'data')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.tools import mesh_faces_to_triangles
dashed_line = "----------------------------------------------------------------\n"
# ==============================================================================
# File
# ==============================================================================




# total folders (-1 for keypoint folder)
total_folders = len([i for i in os.listdir(dataroot)]) - 1
subdivide = False 


def handle_folder(object_folder, dataroot):

    # remove shards
    folder_path = os.path.join(dataroot, object_folder)
    shard_list = [name for name in os.listdir(folder_path) if 'shard' in name]
    for shard in shard_list:
        os.remove(os.path.join(folder_path, shard))
    
    # remove npy in cleaned
    folder_path = os.path.join(dataroot, object_folder, 'cleaned')
    if not os.path.exists(folder_path):
        return
    npy_list = [name for name in os.listdir(folder_path) if name.endswith('.npy')]
    for file in npy_list:
        os.remove(os.path.join(folder_path, file))
        
    print(f'Processed folder {object_folder}')

Parallel(n_jobs=6)(delayed(handle_folder)(folder, dataroot) for folder in os.listdir(dataroot))
