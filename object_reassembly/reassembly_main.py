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
    fragments = []
    fragment_matches = []
    kpts = {}
    kpt_matches = {}
    transformations = {}

    def __int__(self):
        self.fragments = None
        self.fragment_matches = None
        self.kpts = None
        self.kpt_matches = None
        self.transformations = None

    @classmethod
    def load_object(self, path):
        for object in os.listdir(path):
            if object.endswith('.obj'):
                self.fragments.append(Mesh.from_obj(path+object))

        matches_path = os.listdir(path+"matching/")[1]
        self.fragment_matches = np.load(path+"matching/"+matches_path)

        for idx, points in enumerate(os.listdir(path+"keypoints/")):
            self.kpts[idx] = np.load(path+"keypoints/"+points)

    @classmethod
    def load_kpt_matches(self, matches0, matches1, idx0, idx1):
        self.kpt_matches[(idx0, idx1)] = matches0
        self.kpt_matches[(idx1, idx0)] = matches1

    @classmethod
    def find_transformations_first3kpts(self):
        for fragment0 in range(len(self.fragments)-1):
            for fragment1 in range(len(self.fragments)-1):
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



# def load_keypoints():
#     kpts0 = np.load(path + "keypoints/bottle_10_seed_1_kpts_0.npy")
#     kpts1 = np.load(path + "keypoints/bottle_10_seed_1_kpts_2.npy")
#     return kpts0, kpts1


# def get_transformation(kpts0, kpts1, matches0, matches1) -> np.array:
#     pts0 = np.array([kpts0[idx] for idx in matches0[0:3]])
#     pts1 = np.array([kpts1[idx] for idx in matches1[0:3]])
#
#     centroid0 = np.mean(pts0, axis=1)
#     centroid1 = np.mean(pts1, axis=1)
#
#     centroid0 = centroid0.reshape(-1, 1)
#     centroid1 = centroid1.reshape(-1, 1)
#
#     m0 = pts0 - centroid0
#     m1 = pts1 - centroid1
#
#     H = m0 @ np.transpose(m1)
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#
#     t = -R @ centroid0 + centroid1
#
#     return R, t


if __name__ == "__main__":

    viewer = App()

    # kpts0, kpts1, matches0, matches1 = load_data()
    # kpts0, kpts1 = load_keypoints()
    dummy_matches0 = [0, 1, 2]
    dummy_matches1 = [0, 1, 2]

    bottle = FracturedObject()

    bottle.load_object(path)
    for i in range(10):
        for j in range(10):
            if bottle.fragment_matches[i][j]:
                bottle.load_kpt_matches(dummy_matches0, dummy_matches1, i, j)
    bottle.find_transformations_first3kpts()


    # r, t = get_transformation(kpts0, kpts1, matches0, matches1)

    # pc_1 = load_example_data()
    # viewer.add(pc_1)

    viewer.run()


    # Mesh.transform(pc_1, R)
    # Mesh.transform(pc_1, T)
    #
    # viewer.add(pc_1)

    viewer.run()

