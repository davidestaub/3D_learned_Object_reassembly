import os
import numpy as np
from tools import *

from compas.datastructures import Mesh
from compas.datastructures import mesh_explode

# ==============================================================================
# File
# ==============================================================================
ROOT = select_folder()
CLEANED = os.path.join(ROOT, 'cleaned\\')
SUBDV = os.path.join(ROOT, 'subdv\\')

# create a directory for cleaned files if not yet existing
os.chdir(ROOT)
if not os.path.isdir('cleaned'):
    os.mkdir('cleaned')


# ==============================================================================
# Output
# ==============================================================================
object_name = ROOT.split('\\')[-1]

for i, filename in enumerate(os.listdir(ROOT)):
    if filename.endswith(".obj"):
        FILE_I = os.path.join(ROOT, filename)
        if "shard" in filename: 
            mesh = Mesh.from_obj(FILE_I)

            print(i, filename)
            # explode joined meshes
            exploded_meshes = mesh_explode(mesh)
            if len(exploded_meshes) > 1:
                print("The object", filename, "has",len(exploded_meshes), " loose parts!")
                print("Sepparating parts...")
            
            for ex_mesh in exploded_meshes:
                FILE_O_NPY = os.path.join(CLEANED, '%s_%s.npy' % (object_name, i))
                FILE_O_OBJ = os.path.join(CLEANED, '%s_%s.obj' % (object_name, i))

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

    elif filename.endswith(".mtl"):
        FILE_E = os.path.join(ROOT, filename)
        os.remove(FILE_E)
