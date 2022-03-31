import os
from tkinter.ttk import Progressbar
from compas.datastructures import Mesh
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg
from compas.utilities import i_to_rgb
from tools import printProgressBar
from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__)))

data_list = os.listdir(os.path.join(HERE, 'data'))
print("id  name")
for idx, val in enumerate(data_list):
    print(idx," ", val)

idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
SUBFOLDER = data_list[idx]

FILE_FOLDER = os.path.join(HERE, 'data', SUBFOLDER)
FILE_FOLDER_CLEANED = os.path.join(HERE, 'data', SUBFOLDER, 'cleaned')
print("Opening folder:", FILE_FOLDER)


# initialize viewer
viewer = App()

file_nums = len(os.listdir(FILE_FOLDER_CLEANED))/2
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
