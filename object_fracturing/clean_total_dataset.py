from hashlib import new
from importlib.resources import path
from msilib.schema import Class
import os
import numpy as np
from tools import *
import copy
import logging
from joblib import Parallel, delayed

from compas.datastructures import Mesh
from compas.datastructures import mesh_explode
from compas.datastructures import mesh_subdivide_corner

# ==============================================================================
# File
# ==============================================================================
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
dataroot = os.path.join(here, 'data')

# total folders (-1 for keypoint folder)
total_folders = len([i for i in os.listdir(dataroot)]) - 1

class Progress:
    def __init__(self) -> None:
        self.counter = 0
    def update(self):
        self.counter += 1
        print(self.counter)
        printProgressBar(self.counter, total_folders, prefix="Total progress:")

progress = Progress()  


def handle_folder(object_folder, dataroot, idx):
    log = []
    # skip keypoints
    if object_folder == 'keypoints':
        return
    
    folder_path = os.path.join(dataroot, object_folder)
    # delete the log file
    if os.path.exists(folder_path + "\\log.txt"):
        os.remove(folder_path + "\\log.txt")
    #cleane the folder
    for filename in os.listdir(folder_path):
        if filename  == 'cleaned' or filename == 'subdv':
            continue
        file_path = os.path.join(folder_path, filename)
        # delete material list files
        if filename.endswith('.mtl'):
            os.remove(file_path)
        #if it's a shard, rename them uniformly
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

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if "shard" in filename: 
            mesh = Mesh.from_obj(file_path)

            # explode them to sepparate loose parts
            exploded_meshes = mesh_explode(mesh)
            if len(exploded_meshes) > 1:
                log.append("The object:     " + filename + " had " + str(len(exploded_meshes))+ " loose parts!\n")

            # save all the individual objects
            for ex_mesh in exploded_meshes:
                # delete tiny pieces
                if len(list(ex_mesh.vertices())) < 100:
                    continue
                # save to new file
                name = object_folder + "_cleaned." + str(shard_counter) + ".obj"
                FILE_O_OBJ = os.path.join(folder_path,'cleaned',name)
                shard_counter += 1
                ex_mesh.to_obj(FILE_O_OBJ)

    # subdivide the mesh sizes to be approximately the same for each piece
    # else usip could focus more towards dense regions

    # check if the cleaned folder exists yet
    os.chdir(folder_path)
    if not os.path.isdir('subdv'):
        os.mkdir('subdv')

    # get all the meshes for the whole object in each shard 
    meshes = []
    max_len_v = 2000
    cleaned_path = os.path.join(folder_path, 'cleaned')

    for filename in os.listdir(cleaned_path):
        if filename.endswith(".obj"):
            filepath = os.path.join(cleaned_path, filename)
            mesh = Mesh.from_obj(filepath)
            len_v = len(list(mesh.vertices()))
            if len_v > max_len_v:
                max_len_v = len_v
            meshes.append(mesh)

    subdv_path = os.path.join(folder_path, 'subdv')
    # save the meshes after they are "downsampled"
    avg_faces = []
    avg_vertices = []
    avg_edges = []
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

        while len_v < max_len_v / 2:
            mesh = mesh_subdivide_corner(mesh, k=1)
            len_v = len(list(mesh.vertices()))

        try:
            vertices = np.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
            normals = np.array([mesh.vertex_normal(vkey) for vkey in mesh.vertices()])
            datas = np.concatenate((vertices, normals), axis=1)
            np.save(FILE_O + ".npy", datas)
            mesh.to_obj(FILE_O+".obj")
        except Exception as e:
            log.append("ERROR AT FILE: " + FILE_O)
            log.append(str(e))
            log.append('\n')

    # append averages
    log.append("Average number of faces: " + str(np.average(avg_faces)) + "\n")
    log.append("Average number of vertices: " + str(np.average(avg_vertices)) + "\n")
    log.append("Average number of edges: " + str(np.average(avg_edges)) + "\n")
    log_path = os.path.join(folder_path, 'log.txt')

    with open(log_path, "w+") as text_file:
        text_file.write(''.join(log))
    # update the progres bar
    printProgressBar(idx, total_folders, prefix="Total Progress:")

Parallel(n_jobs=6)(delayed(handle_folder)(folder, dataroot, idx) for idx, folder in enumerate(os.listdir(dataroot)))