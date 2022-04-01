import os
from secrets import choice
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from tools import *
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg

from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
ROOT = select_folder()
CLEANED = os.path.join(ROOT, 'cleaned\\')
SUBDV = os.path.join(ROOT, 'subdv\\')
FOLDER = ROOT

print(1, "normal")
print(2, "cleaned")
print(3, "subdivided")
choice = int(input("Which version do you want to plot?\n"))
if choice not in [1,2,3]:
    exit("You failed to type an integer between 1 and 3...\n\n\n")

if choice == 1:
    FOLDER = ROOT
if choice == 2:
    FOLDER = CLEANED
if choice == 3:
    FOLDER = SUBDV

if not os.path.exists(FOLDER):
    exit("There is no matching folder")

explode = bool(int(input("Do you want do explode the data?\n1:Yes\n0:No\n")))

# ==============================================================================
# Loading the data
# ==============================================================================

# initialize viewer
viewer = App()

files = [ob for ob in os.listdir(FOLDER) if ob.endswith(".obj")]
file_nums = len(files)
print("There are", file_nums, "of fragments.")

counter = 0
if choice == 1:
    counter += 1

if (explode):
    for filename in os.listdir(ROOT):
        FILE_I = os.path.join(ROOT, filename)
        if filename.endswith(".obj") and "shard" not in filename:
            mesh = Mesh.from_obj(FILE_I)
            mass_center = mesh.centroid()

    for filename in os.listdir(FOLDER):
        if filename.endswith(".obj"):
            if choice == 1 and "shard" not in filename:
                continue
            FILE_I = os.path.join(FOLDER, filename)
            printProgressBar(counter+1,file_nums, prefix="Loading Fragments")
            mesh = Mesh.from_obj(FILE_I)
            mesh_center = mesh.centroid()
            vec = cg.Vector(*[a - b for (a, b) in zip(mesh_center, mass_center)])
            vec = vec * 0.5
            T = cg.Translation.from_vector(vec)
            mesh_transform_numpy(mesh, T)
            viewer.add(mesh, facecolor=i_to_rgb(counter/file_nums, True))
            counter += 1
else:
    for filename in files:
        # skip the whole piece if the original data
        # is plotted
        if choice == 1 and "shard" not in filename:
            continue
        printProgressBar(counter+1,file_nums, prefix="Loading Fragments")
        FILE_I = os.path.join(FOLDER, filename)
        mesh = Mesh.from_obj(FILE_I)
        len_vertices = len(list(mesh.vertices()))

        viewer.add(mesh, facecolor=i_to_rgb(counter/file_nums, True))
        counter += 1

# ==============================================================================
# Viz
# ==============================================================================
viewer.run()