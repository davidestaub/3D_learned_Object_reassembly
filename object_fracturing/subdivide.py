import os
import copy
import numpy as np
import math as m
from tools import *

from compas.datastructures import Mesh
from compas.datastructures import mesh_subdivide_corner

from compas_view2.app import App
# ==============================================================================
# File
# ==============================================================================
ROOT = select_folder()
CLEANED = os.path.join(ROOT, 'cleaned\\')
SUBDV = os.path.join(ROOT, 'subdv\\')

# create a directory for cleaned files if not yet existing
os.chdir(ROOT)
if not os.path.isdir('subdv'):
    os.mkdir('subdv')

# ==============================================================================
# Analysis Data
# ==============================================================================

meshes = []
max_len_v = 2000

for filename in os.listdir(CLEANED):
    if filename.endswith(".obj"):
        filepath = os.path.join(CLEANED, filename)
        mesh = Mesh.from_obj(filepath)
        len_v = len(list(mesh.vertices()))
        if len_v > max_len_v:
            max_len_v = len_v
        meshes.append(mesh)

print("There are", len(meshes), "fragments loaded.")
print("Maximum vertice length:", max_len_v)

# ==============================================================================
# Subdivide and Output
# ==============================================================================
object_name = ROOT.split('\\')[-1]
problem_ind = []

for i, mesh in enumerate(meshes):
    FILE_O = os.path.join(SUBDV, '%s_%s' % (object_name, i))
    len_f = len(list(mesh.faces()))
    len_v = len(list(mesh.vertices()))
    len_e = len(list(mesh.edges()))

    while len_v < max_len_v / 2:
        mesh = mesh_subdivide_corner(mesh, k=1)
        len_v = len(list(mesh.vertices()))

    print(max_len_v, len_v)
    try:
        vertices = np.array([mesh.vertex_coordinates(vkey)
                            for vkey in mesh.vertices()])
        normals = np.array([mesh.vertex_normal(vkey)
                           for vkey in mesh.vertices()])
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
