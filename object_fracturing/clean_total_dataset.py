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
    log = []

    folder_path = os.path.join(dataroot, object_folder)

    # delete the premade cleaned and subdv folder
    try:
        shutil.rmtree(os.path.join(folder_path, 'cleaned'))
    except:
        pass

    # recreate directions
    os.chdir(folder_path)
    os.mkdir('cleaned')

    # delete the log file
    if os.path.exists(folder_path + "\\log.txt"):
        os.remove(folder_path + "\\log.txt")
    
    # clean the folder
    for filename in os.listdir(folder_path):
        if filename in ['keypoints', 'keypoints_harris']:
            try:
                shutil.rmtree(os.path.join(folder_path, filename))
            except:
                pass
            continue
        if filename in ['cleaned', 'processed']:
            continue
        file_path = os.path.join(folder_path, filename)
        # delete material list files
        if filename.endswith('.mtl'):
            os.remove(file_path)
        # if it's a shard, rename them uniformly
        elif 'shard' in filename:
            # now rename the shards
            if filename.endswith('.obj'):
                # detect shard number
                shard_split = filename.split('.')[-2]
                shard_number = 'XXX'
                # there are either numbers or the name (0th shard)
                # convert them to the number
                if '_' in shard_split:
                    shard_number = '000'
                else:
                    shard_number = shard_split
                new_name = object_folder + "_shard." + shard_number + ".obj"
            if filename.endswith('.ply'):
                # detect shard number
                shard_number = filename.split('.')[0].split('_')[-1]
                new_name = object_folder + "_shard." + shard_number + ".ply"
            
            new_name = os.path.join(folder_path, new_name)
            try:
                os.rename(file_path, new_name)
            except:
                pass
        else:
            new_name = os.path.join(folder_path, object_folder)
            if filename.endswith('.obj'):
                new_name = new_name + ".obj"
            if filename.endswith('.ply'):
                new_name = new_name + ".ply"
            try:
                os.rename(file_path, new_name)
            except:
                pass

    # now that the folder is cleaned, one can create the cleaned data and subdidive the meshes
    # check if the cleaned folder exists yet
    os.chdir(folder_path)
    if not os.path.isdir('cleaned'):
        os.mkdir('cleaned')

    # read the new files and clean the pointclouds
    shard_counter = 0
    piece_counter = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if "shard" in filename:
            if filename.endswith('.obj'):
                mesh = Mesh.from_obj(file_path)

            if filename.endswith('.ply'):
                mesh = Mesh.from_ply(file_path)
                mesh.make

            # explode them to sepparate loose parts
            exploded_meshes = mesh_explode(mesh)
            if len(exploded_meshes) > 1:
                log.append("The object:     " + filename + " had " +
                           str(len(exploded_meshes)) + " loose parts!\n")

            # save all the individual objects
            for ex_mesh in exploded_meshes:
                # delete tiny pieces
                if len(list(ex_mesh.vertices())) < 1000:
                    log.append(f'Deleted a small fragment of shard: {shard_counter}\n')
                    continue

                # center to origin (this destroys matching!!)
                #center = ex_mesh.centroid()
                #vec =  cg.Vector(-center[0], -center[1], -center[2])
                #mesh_transform_numpy(ex_mesh, cg.Translation.from_vector(vec))
                # save to new file
                name = object_folder + "_cleaned." + str(piece_counter)
                FILE_NPY = os.path.join(folder_path, 'cleaned', name + ".npy")
                FILE_OBJ = os.path.join(folder_path, 'cleaned', name + ".obj")
                
                ex_mesh.to_obj(FILE_OBJ)
                piece_counter += 1
                vertices = np.array([ex_mesh.vertex_coordinates(vkey)for vkey in ex_mesh.vertices()])
                normals = np.array([ex_mesh.vertex_normal(vkey)for vkey in ex_mesh.vertices()])
                data = np.concatenate((vertices, normals), axis=1)
                np.save(FILE_NPY, data)

            shard_counter += 1

    log_path = os.path.join(folder_path, 'log.txt')

    with open(log_path, "w+") as text_file:
        text_file.write(''.join(log))
    print(f'Processed folder {object_folder}')

Parallel(n_jobs=12)(delayed(handle_folder)(folder, dataroot) for folder in os.listdir(dataroot))
