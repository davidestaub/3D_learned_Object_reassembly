import os
import numpy as np
import shutil
from joblib import Parallel, delayed
from compas.datastructures import Mesh
from compas.datastructures import mesh_explode
import open3d as o3d
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
dataroot = os.path.join(here, 'data_full', 'data')
dashed_line = "----------------------------------------------------------------\n"


# total folders (-1 for keypoint folder)
total_folders = len([i for i in os.listdir(dataroot)])
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
        if filename in ['cleaned', 'processed']:
            continue
        file_path = os.path.join(folder_path, filename)
        # delete material list files
        if filename.endswith('.mtl'):
            os.remove(file_path)

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
                if not mesh.is_manifold():
                    print(f'Mesh {filename} not manifold!')
                    return
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

                # save to new file
                name = object_folder + "_cleaned." + str(piece_counter)
                FILE_PCD = os.path.join(folder_path, 'cleaned', name + ".pcd")
                FILE_OBJ = os.path.join(folder_path, 'cleaned', name + ".obj")
                ex_mesh.to_obj(FILE_OBJ)
                piece_counter += 1
                vertices = np.array([ex_mesh.vertex_coordinates(vkey)for vkey in ex_mesh.vertices()], dtype=np.float32)
                normals = np.array([ex_mesh.vertex_normal(vkey)for vkey in ex_mesh.vertices()], dtype=np.float32)
                # create a pointcloud and save as pcd file with o3d 
                pcd = o3d.geometry.PointCloud()
                pcd.normals = o3d.utility.Vector3dVector(normals)
                pcd.points= o3d.utility.Vector3dVector(vertices)
                o3d.io.write_point_cloud(FILE_PCD, pcd)


            shard_counter += 1

    log_path = os.path.join(folder_path, 'log.txt')

    with open(log_path, "w+") as text_file:
        text_file.write(''.join(log))
    print(f'Processed folder {object_folder}')

folders = os.listdir(dataroot)
Parallel(n_jobs=4)(delayed(handle_folder)(folder, dataroot) for folder in folders)
