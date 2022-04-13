import os
import open3d
import compas
from tools import *
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from compas.geometry import Translation, Rotation
from compas_view2.app import App

path = "../data/bottle_10_seed_1/"


def load_example_data():
    data = Mesh.from_obj(path + "bottle_10_seed_1_shard.000.obj")

    return data


def load_keypoints():
    kpts0 = np.load(path + "keypoints/bottle_10_seed_1_kpts_0.npy")
    kpts1 = np.load(path + "keypoints/bottle_10_seed_1_kpts_2.npy")
    return kpts0, kpts1


def get_transformation(kpts0, kpts1, matches0, matches1) -> np.array:
    pts0 = np.array([kpts0[idx] for idx in matches0[0:3]])
    pts1 = np.array([kpts1[idx] for idx in matches1[0:3]])

    centroid0 = np.mean(pts0, axis=1)
    centroid1 = np.mean(pts1, axis=1)

    centroid0 = centroid0.reshape(-1, 1)
    centroid1 = centroid1.reshape(-1, 1)

    m0 = pts0 - centroid0
    m1 = pts1 - centroid1

    H = m0 @ np.transpose(m1)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid0 + centroid1

    return R, t


if __name__ == "__main__":

    viewer = App()

    # kpts0, kpts1, matches0, matches1 = load_data()
    kpts0, kpts1 = load_keypoints()
    matches0 = [0, 1, 2]
    matches1 = [0, 1, 2]

    r, t = get_transformation(kpts0, kpts1, matches0, matches1)

    pc_1 = load_example_data()
    viewer.add(pc_1)
    viewer.run()

    T = Translation.from_vector(t)
    col4 = np.array([[0], [0], [0]])
    tmp = np.concatenate((r, col4), axis=1)
    row4 = np.array([0, 0, 0, 1])
    tmp2 = np.append(tmp, row4)
    R = Rotation.from_list(list(tmp2))

    Mesh.transform(pc_1, R)
    Mesh.transform(pc_1, T)

    viewer.add(pc_1)

    viewer.run()

