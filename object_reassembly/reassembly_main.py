import os
import open3d
import compas
from tools import *
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from compas.geometry import Translation, Rotation
from compas_view2.app import App

here = os.path.dirname(os.path.abspath(__file__))
path = "../data/bottle_10_seed_1/"


class FracturedObject:
    def __int__(self):
        self.fragments = []
        self.fragment_matches = []
        self.kpts = {}
        self.kpt_matches = {}
        self.transformations = {}

    def load_object(self, path):
        for object in os.listdir(path):
            if object.endswith('.obj'):
                self.fragments.append(Mesh.from_obj(object))

        matches_path = os.listdir(path)[0]
        self.fragment_matches = np.load(matches_path)

        for idx, points in enumerate(os.listdir(path)):
            self.kpts[idx] = np.load(points)

    def load_kpt_matches(self, matches0, matches1, idx0, idx1):
        self.kpt_matches[(idx0, idx1)] = matches0
        self.kpt_matches[(idx1, idx0)] = matches1

    def find_transformations_first3kpts(self):
        for fragment0 in range(len(self.fragments)):
            for fragment1 in range(len(self.fragments)):
                if self.fragment_matches[fragment0][fragment1]:
                    pts0 = np.array([self.kpts[fragment0][idx] for idx in self.kpt_matches[(fragment0, fragment1)][0:3]])
                    pts1 = np.array([self.kpts[fragment1][idx] for idx in self.kpt_matches[(fragment1, fragment0)][0:3]])

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

                    self.transformations[(fragment0, fragment1)] = (R, t)



def load_example_data() -> list:
    data = []
    for object in os.listdir(here):
        data.append(Mesh.from_obj(object))

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

