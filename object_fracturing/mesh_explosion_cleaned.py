import os
from tkinter.ttk import Progressbar
from compas.datastructures import Mesh
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg
from compas.utilities import i_to_rgb
from tools import *
from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
FILE_FOLDER, FILE_FOLDER_CLEANED = select_folder()

# initialize viewer
viewer = App()

file_nums = len([item for item in os.listdir(FILE_FOLDER_CLEANED) if item.endswith(".obj")])
print("Found", file_nums, "files")

for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    if filename.endswith(".obj") and "shard" not in filename:
        FILE_I = os.path.join(FILE_FOLDER, filename)
        mesh = Mesh.from_obj(FILE_I)
        mass_center = mesh.centroid()

counter = 0

for filename in os.listdir(FILE_FOLDER_CLEANED):
    if filename.endswith(".obj"):
        printProgressBar(counter+1,file_nums, prefix="Loading Fragments")
        FILE_I = os.path.join(FILE_FOLDER_CLEANED, filename)
        mesh = Mesh.from_obj(FILE_I)
        mesh_center = mesh.centroid()
        vec = cg.Vector(*[a - b for (a, b) in zip(mesh_center, mass_center)])
        vec = vec * 0.5
        T = cg.Translation.from_vector(vec)
        mesh_transform_numpy(mesh, T)
        viewer.add(mesh, facecolor=i_to_rgb(counter/file_nums, True))
        counter += 1

# ==============================================================================
# Viz
# ==============================================================================
viewer.run()
