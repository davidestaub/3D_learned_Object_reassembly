import os
from secrets import choice
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from sympy import primitive
from tools import *
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg
from compas.geometry import Point, Sphere, Pointcloud
import numpy as np
from compas_view2.app import App
import open3d as o3d

# ==============================================================================
# File
# ==============================================================================
ROOT = select_folder(mode='keypoints')
# get the data folder
os.chdir(ROOT)
os.chdir('..')
DATAROOT = os.path.join(os.getcwd())
KPTS_IN = os.path.join(ROOT, 'input\\')
KPTS_OUT = os.path.join(ROOT, 'output\\')



# initialize viewer
viewer = App()
explode = False

files_in = [ob for ob in os.listdir(KPTS_IN) if ob.endswith(".npy")]
file_in_nums = len(files_in)
print("There are", file_in_nums, "of input fragments.")

files_out = [ob for ob in os.listdir(KPTS_OUT) if ob.endswith(".npy")]
file_out_nums = len(files_out)
print("There are", file_in_nums, "of output fragments.")

if file_in_nums != file_out_nums:
    exit("There are not the same amount of inputs as outputs!")

if file_in_nums > 1:
    explode = bool(int(input("Do you want do explode the data?\n1:Yes\n0:No\n")))


# find the different file primitives and check if there exists a path
primitives = []

for file in files_in:
    splits = file.split('_')
    file_primitive = '_'.join(splits[:-1])
    file_path = os.path.join(DATAROOT,file_primitive)
    if not os.path.exists(file_path):
        exit("There is no folder for the original object", file_primitive ," in data!")
    primitives.append(file_primitive)

# convert to a set for unique values
primitives = set(primitives)

for object_primitive in primitives:

    counter = 0
    translation_vectors = []
    idx = 0

    if (explode):
        # construct the path to the whole object, located in the dataroot
        ROOT = os.path.join(DATAROOT, object_primitive)
        for filename in os.listdir(ROOT):
            file_obj = os.path.join(ROOT, filename)
            # ignore shards
            if filename.endswith(".obj") and "shard" not in filename:
                mesh = Mesh.from_obj(file_obj)
                mass_center = mesh.centroid()
        # load the whole fragments
        for filename in os.listdir(KPTS_IN):
            # load the objects in the keypoints in folder, only load one object_primitive at once
            if filename.endswith(".obj") and object_primitive in filename:
                FILE_I = os.path.join(KPTS_IN, filename)

                printProgressBar(counter+1,file_in_nums, prefix= "Loading Fragments for primitive: "+ object_primitive)

                # load the obj mesh and add it to the viewer
                mesh = Mesh.from_obj(FILE_I)
                mesh_center = mesh.centroid()
                vec = cg.Vector(*[a - b for (a, b) in zip(mesh_center, mass_center)])
                vec = vec * 0.5

                T = cg.Translation.from_vector(vec)
                translation_vectors.append(T)

                mesh_transform_numpy(mesh, T)
                viewer.add(mesh, facecolor=i_to_rgb(counter/file_in_nums, True))
                counter += 1
        if len(translation_vectors) == 0:
            print("Didn't find any obj files in the input folder!")
            if bool(input("Shall they be searched automatically?\n1:yes\n0:No\n")):
                automatic_in = os.path.join(DATAROOT,object_primitive,'subdv')
                if not os.path.exists(automatic_in):
                    exit("Error: There is no folder:", automatic_in)
                else:
                    for filename in os.listdir(automatic_in):
                        if filename.endswith(".obj"):
                            file_in = os.path.join(automatic_in, filename)

                            printProgressBar(counter+1,file_in_nums, prefix= "Loading Fragments for primitive: "+ object_primitive)

                            # load the obj mesh and add it to the viewer
                            mesh = Mesh.from_obj(file_in)
                            mesh_center = mesh.centroid()
                            vec = cg.Vector(*[a - b for (a, b) in zip(mesh_center, mass_center)])
                            vec = vec * 0.5

                            T = cg.Translation.from_vector(vec)
                            translation_vectors.append(T)

                            mesh_transform_numpy(mesh, T)
                            viewer.add(mesh, facecolor=i_to_rgb(counter/file_in_nums, True))
                            counter += 1

            else:
                exit("Abort")


        # load the keypoints
        counter = 0
        for filename in os.listdir(KPTS_OUT):
            # load the objects in the keypoints in folder
            if filename.endswith(".npy") and object_primitive in filename:
                FILE_I = os.path.join(KPTS_OUT, filename)

                printProgressBar(counter+1,file_out_nums, prefix= "Loading Pointcloud for primitive: "+ object_primitive)
                
                # load the keypoints without saliency score
                kpts = np.load(FILE_I)

                
                points = []
                # create a small sphere for each keypoint
                for point in kpts:
                    points.append(Point(point[0], point[1], point[2]))

                # generate a pointcloud
                cloud = Pointcloud(points)

                # transform the cloud the same way as the corresponding piece
                cloud_center = cloud.centroid
                cloud.transform(translation_vectors[idx])

                viewer.add(cloud,facecolor=i_to_rgb(0.5, True))
                counter += 1
                idx += 1
    else:

        # load the whole fragments
        print(KPTS_IN)
        print(object_primitive)
        for filename in os.listdir(KPTS_IN):
            print(filename)
            # load the objects in the keypoints in folder, only load one object_primitive at once
            if filename.endswith(".obj") and object_primitive in filename:
                FILE_I = os.path.join(KPTS_IN, filename)

                printProgressBar(counter+1,file_in_nums, prefix= "Loading Fragments for primitive: "+ object_primitive)

                # load the obj mesh and add it to the viewer
                mesh = Mesh.from_obj(FILE_I)
                viewer.add(mesh, facecolor=i_to_rgb(counter/file_in_nums, True))
                counter += 1
        # automatic search if needed
        if counter == 0:
            print("Didn't find any obj files in the input folder!")
            if bool(input("Shall they be searched automatically?\n1:yes\n0:No\n")):
                automatic_in = os.path.join(DATAROOT,object_primitive,'subdv')
                if not os.path.exists(automatic_in):
                    exit("Error: There is no folder:", automatic_in)
                else:
                    for filename in os.listdir(automatic_in):
                        if filename.endswith(".obj"):
                            file_in = os.path.join(automatic_in, filename)

                            printProgressBar(counter+1,file_in_nums, prefix= "Loading Fragments for primitive: "+ object_primitive)

                            # load the obj mesh and add it to the viewer
                            mesh = Mesh.from_obj(file_in)
                            mesh_center = mesh.centroid()

                            viewer.add(mesh, facecolor=i_to_rgb(counter/file_in_nums, True))
                            counter += 1

            else:
                exit("Abort")

        # load the keypoints
        counter = 0
        for filename in os.listdir(KPTS_OUT):
            # load the objects in the keypoints in folder
            if filename.endswith(".npy") and object_primitive in filename:
                FILE_I = os.path.join(KPTS_OUT, filename)

                printProgressBar(counter+1,file_out_nums, prefix= "Loading Keypoints for primitive"+ object_primitive)
                
                # load the keypoints without saliency score
                kpts = np.load(FILE_I)

                points = []
                scores = []

                # create a small sphere for each keypoint
                for point in kpts:
                    scores.append(point[3])
                    points.append([point[0], point[1], point[2]])
                scores = scores / max(scores)
                # filter scores
                #filter = np.where(scores > 0.5)
                points = np.array(points)
                print(len(points))
                points = points[scores > 0.8]
                print(len(points))
                #points = [points[i] for i in filter]

                points_compas = []
                for item in points:
                    points_compas.append(Point(item[0],item[1],item[2]))

                # generate a pointcloud
                cloud = Pointcloud(points_compas)
                viewer.add(cloud, color=i_to_rgb(1, True))
                counter += 1

    viewer.run()