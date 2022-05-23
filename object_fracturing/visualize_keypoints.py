import os
import open3d as o3d
import numpy as np
from compas_view2.app import App
from compas.geometry import Pointcloud, Translation
from compas. datastructures import Mesh, mesh_transform_numpy

def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres

if __name__ == '__main__':
    
    # chose a data folder
    folder = "data"
    here = os.path.dirname(os.path.abspath(__file__))
    data_list = os.listdir(os.path.join(here, folder))


    print("id  name")
    for idx, val in enumerate(data_list):
        print(idx, " ", val)
    idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
    subfolder = data_list[idx]
    ROOT = os.path.join(here, folder, subfolder)

    # get the other folders
    CLEANED = os.path.join(ROOT, 'cleaned')
    KPTS_IN = os.path.join(ROOT,'processed','keypoints')

    kpts_mode = 'SD'

    # chose a fragments
    data_list = [i for i in os.listdir(CLEANED) if i.endswith('pcd')]
    print("id  name")
    for idx, val in enumerate(data_list):
        if val.endswith('.pcd'):
            print(idx, " ", val)
    idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
    file = data_list[idx]
    idx = int(file.split('cleaned.')[1].split('.')[0])
    print(file, idx)

    # extract the corresponding filename
    kpts_mode = 'hybrid'
    kpts_file = f'keypoints_{kpts_mode}.{idx}.npy'
    kpts = np.load(os.path.join(KPTS_IN, kpts_file))[:,:3]
    kpts_pcd = o3d.geometry.PointCloud()
    kpts_pcd.points= o3d.utility.Vector3dVector(kpts)
    compas_kpts= Pointcloud(kpts_pcd.points)


    center = o3d.io.read_point_cloud(os.path.join(CLEANED, file)).get_center()

    viewer = App(show_grid=False, viewmode='lighted')
    compas_mesh = Mesh().from_obj(os.path.join(CLEANED, file.replace('pcd', 'obj')))
    t = Translation().from_vector(center)
    t.invert()
    color_kasia = [254/255, 74/255, 25/255]

    mesh_transform_numpy(compas_mesh, t)
    viewer.add(compas_kpts, color=color_kasia)
    viewer.add(compas_mesh, facecolor=[0.6,0.6,0.6])
    viewer.show()