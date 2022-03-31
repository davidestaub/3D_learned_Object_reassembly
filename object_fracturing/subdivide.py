import os
import copy
import numpy as np
import math as m 
from tools import *

from compas.datastructures import Mesh
from compas.datastructures import mesh_subdivide_corner
from compas.datastructures import mesh_explode

from compas_view2.app import App
from compas.utilities import i_to_rgb
# ==============================================================================
# File
# ==============================================================================
FILE_FOLDER, FILE_FOLDER_CLEANED = select_folder()

# ==============================================================================
# Analysis Data
# ==============================================================================

meshes = []
max_len_v = 2000

for filename in os.listdir(FILE_FOLDER_CLEANED):
    if filename.endswith(".obj"):
        filepath = os.path.join(FILE_FOLDER_CLEANED, filename)
        mesh = Mesh.from_obj(filepath)
        len_v = len(list(mesh.vertices()))
        if len_v > max_len_v:
            max_len_v = len_v
        meshes.append(mesh)

print("There are", len(meshes), "fragments loaded.")
print("Maximum vertice length:", max_len_v)

meshes_copy = copy.deepcopy(meshes)

# ==============================================================================
# Subdivide and Output
# ==============================================================================
os.chdir(FILE_FOLDER)
if not os.path.isdir('subdv'):
    os.mkdir('subdv')

cleared_suffix = FILE_FOLDER.split('\\')[-1].split('_')[0] + "subdv"
problem_ind = []

for i, mesh in enumerate(meshes):
    FILE_O = FILE_FOLDER + "\\subdv\\" + cleared_suffix + "_" + str(i)
    len_f = len(list(mesh.faces()))
    len_v = len(list(mesh.vertices()))
    len_e = len(list(mesh.edges()))

    while len_v < max_len_v / 2:
        mesh = mesh_subdivide_corner(mesh, k=1)
        len_v = len(list(mesh.vertices()))

    print(max_len_v, len_v)
    try:
        vertices = np.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        normals = np.array([mesh.vertex_normal(vkey) for vkey in mesh.vertices()])
        datas = np.concatenate((vertices, normals), axis=1)
        np.save(FILE_O + ".npy", datas)
        mesh.to_obj(FILE_O+".obj")
    except:
        print("soemthing goes wrong...", i)
        problem_ind.append(i)

print(problem_ind)


# # ==============================================================================
# # Viz
# # ==============================================================================
# viewer = App()

# # for i, mesh in enumerate(meshes):
# #     viewer.add(mesh, facecolor=i_to_rgb(i/len(meshes), True))


# for i in problem_ind:
#     viewer.add(meshes_copy[i], facecolor=i_to_rgb(i/len(meshes_copy), True))

# viewer.run()





