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
        shutil.rmtree(os.path.join(folder_path, 'cleaned\\'))
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
        if filename == 'keypoints':
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
            # detect shard number
            shard_split = filename.split('.')[-2]
            shard_number = 'XXX'
            # there are either numbers or the name (0th shard)
            # convert them to the number
            if '_' in shard_split:
                shard_number = '000'
            else:
                shard_number = shard_split
            # now rename the shards
            new_name = object_folder + "_shard." + shard_number + ".obj"
            new_name = os.path.join(folder_path, new_name)
            os.rename(file_path, new_name)
        else:
            new_name = os.path.join(folder_path, object_folder)
            new_name = new_name + ".obj"
            os.rename(file_path, new_name)

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
            mesh = Mesh.from_obj(file_path)

            # explode them to sepparate loose parts
            exploded_meshes = mesh_explode(mesh)
            if len(exploded_meshes) > 1:
                log.append("The object:     " + filename + " had " +
                           str(len(exploded_meshes)) + " loose parts!\n")

            # save all the individual objects
            for ex_mesh in exploded_meshes:
                # delete tiny pieces
                if len(list(ex_mesh.vertices())) < 500:
                    log.append(f'Deleted a small fragment of shard: {shard_counter}\n')
                    continue

                # center to origin
                center = ex_mesh.centroid()
                vec =  cg.Vector(-center[0], -center[1], -center[2])
                mesh_transform_numpy(ex_mesh, cg.Translation.from_vector(vec))
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

    if subdivide:
        # check if the cleaned folder exists yet
        os.chdir(folder_path)
        if not os.path.isdir('subdv'):
            os.mkdir('subdv')

        # get all the meshes for the whole object in each shard
        meshes = []
        max_len_v = 1000
        cleaned_path = os.path.join(folder_path, 'cleaned')

        for filename in os.listdir(cleaned_path):
            if filename.endswith(".obj"):
                filepath = os.path.join(cleaned_path, filename)
                mesh = Mesh.from_obj(filepath)
                len_v = len(list(mesh.vertices()))
                if len_v > max_len_v:
                    max_len_v = len_v
                meshes.append(mesh)

        # construct subdivided pieces path
        subdv_path = os.path.join(folder_path, 'subdv')

        # lists to measure several mesh averages before and after the subdivision
        avg_faces = []
        avg_vertices = []
        avg_edges = []
        avg_faces_after = []
        avg_vertices_after = []
        avg_edges_after = []

        # go through all meshes
        for i, mesh in enumerate(meshes):
            name = object_folder + '_subdv.' + str(i)
            FILE_O = os.path.join(subdv_path, name)
            len_f = len(list(mesh.faces()))
            len_v = len(list(mesh.vertices()))
            len_e = len(list(mesh.edges()))
            # append to the avg counter
            avg_faces.append(len_f)
            avg_vertices.append(len_v)
            avg_edges.append(len_e)

            # upsample the pointcloud such that the smallest mesh has
            # at least half as many vertices as the biggest one
            while len_v < max_len_v / 2:
                mesh = mesh_subdivide_corner(mesh, k=1)
                len_v = len(list(mesh.vertices()))

            try:
                vertices = np.array([mesh.vertex_coordinates(vkey)
                                    for vkey in mesh.vertices()])
                normals = np.array([mesh.vertex_normal(vkey)
                                for vkey in mesh.vertices()])
                datas = np.concatenate((vertices, normals), axis=1)
                np.save(FILE_O + ".npy", datas)
                mesh.to_obj(FILE_O+".obj")
            except Exception as e:
                log.append("ERROR AT FILE: " + FILE_O)
                log.append(str(e))
                log.append('\n')
            # after division log
            len_f = len(list(mesh.faces()))
            len_v = len(list(mesh.vertices()))
            len_e = len(list(mesh.edges()))
            # append to the avg counter
            avg_faces_after.append(len_f)
            avg_vertices_after.append(len_v)
            avg_edges_after.append(len_e)

        # append averages
        log.append(dashed_line)
        log.append("Average number of faces before subdv: " +
                str(np.average(avg_faces)) + "\n")
        log.append("Average number of vertices before subdv: " +
                str(np.average(avg_vertices)) + "\n")
        log.append("Minimal number of vertices before subdv: " +
                str(np.min(avg_vertices)) + "\n")
        log.append("Maximum number of vertices before subdv: " +
                str(np.max(avg_vertices)) + "\n")
        log.append("Average number of edges before subdv: " +
                str(np.average(avg_edges)) + "\n")
        log.append(dashed_line)
        log.append("Average number of faces after subdv: " +
                str(np.average(avg_faces_after)) + "\n")
        log.append("Average number of vertices after subdv: " +
                str(np.average(avg_vertices_after)) + "\n")
        log.append("Minimal number of vertices after subdv: " +
                str(np.min(avg_vertices_after)) + "\n")
        log.append("Maximum number of vertices after subdv: " +
                str(np.max(avg_vertices_after)) + "\n")
        log.append("Average number of edges after subdv: " +
                str(np.average(avg_edges_after)) + "\n")

    log_path = os.path.join(folder_path, 'log.txt')

    with open(log_path, "w+") as text_file:
        text_file.write(''.join(log))
    print(f'Processed folder {object_folder}')

Parallel(n_jobs=12)(delayed(handle_folder)(folder, dataroot) for folder in os.listdir(dataroot))
