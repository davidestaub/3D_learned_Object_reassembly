import os
import numpy as np

from compas.datastructures import Mesh
from compas.datastructures import mesh_explode
from compas.files import OBJWriter

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


# create a directory for cleaned files if not yet existing
os.chdir(FILE_FOLDER)
if not os.path.isdir('cleaned'):
    os.mkdir('cleaned')


# ==============================================================================
# Output
# ==============================================================================
counter = 0
for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    if filename.endswith(".obj"):
        FILE_I = os.path.join(FILE_FOLDER, filename)
        if "shard" in filename: 
            mesh = Mesh.from_obj(FILE_I)

            print(i, filename)
            # explode joined meshes
            exploded_meshes = mesh_explode(mesh)
            if len(exploded_meshes) > 1:
                print("The object", filename, "has",len(exploded_meshes), " loose parts!")
                print("Sepparating parts...")
            
            for ex_mesh in exploded_meshes:
                FILE_O_NPY = os.path.join(FILE_FOLDER,'cleaned', '%s_%s.npy' % (SUBFOLDER, counter))
                FILE_O_OBJ = os.path.join(FILE_FOLDER,'cleaned', '%s_%s.obj' % (SUBFOLDER, counter))

                # delete tiny pieces
                if len(list(ex_mesh.vertices())) < 100:
                    print("Small fragment deleted.")
                    continue
                ex_mesh.to_obj(FILE_O_OBJ)

                vertices = np.array([ex_mesh.vertex_coordinates(vkey) for vkey in ex_mesh.vertices()])
                normals = np.array([ex_mesh.vertex_normal(vkey) for vkey in ex_mesh.vertices()])

                datas = np.concatenate((vertices, normals), axis=1)
                #print(np.shape(datas))
                np.save(FILE_O_NPY, datas)

                counter += 1

    elif filename.endswith(".mtl"):
        FILE_E = os.path.join(FILE_FOLDER, filename)
        os.remove(FILE_E)





