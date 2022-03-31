import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb

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

FILE_FOLDER = os.path.join(HERE, 'data', SUBFOLDER, 'cleaned')
print("Opening folder:", FILE_FOLDER)

# initialize viewer
viewer = App()

files = [ob for ob in os.listdir(FILE_FOLDER) if ob.endswith(".obj")]
file_nums = len(files)
print("There are", file_nums, "of fragments.")

# calculate the total vertices
total_vertices = 0
for i, filename in enumerate(files):
    FILE_I = os.path.join(FILE_FOLDER, filename)
    mesh = Mesh.from_obj(FILE_I)
    len_vertices = len(list(mesh.vertices()))
    total_vertices += len_vertices
    viewer.add(mesh, facecolor=i_to_rgb(i/file_nums, True))


print("Total fracture vertices:", total_vertices)

# ==============================================================================
# Viz
# ==============================================================================
viewer.run()