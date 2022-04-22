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

                    row = np.array([[0, 0, 0, 1]])
                    tmp = np.concatenate((R, t), axis=1)

                    T = np.concatenate((tmp, row))

                    self.transformations[(fragment0, fragment1)] = T


def main():
    viewer = App()

    dummy_matches0 = [0, 1, 2]
    dummy_matches1 = [0, 1, 2]

    bottle = FracturedObject()

    bottle.load_object(path)
    for i in range(10):
        for ii in range(10):
            if bottle.fragment_matches[i][ii]:
                bottle.load_kpt_matches(dummy_matches0, dummy_matches1, i, ii)
    bottle.find_transformations_first3kpts()

    matched = bool(0)
    for i in range(10):
        matched = 0
        ii = i
        while not matched:
            if bottle.fragment_matches[i][ii]:
                T = bottle.transformations[(i, ii)]
                Mesh.transform(bottle.fragments[i], T)
                viewer.add(bottle.fragments[i])
                matched = 1
            ii += 1
            if ii >= len(bottle.fragments):
                break

    viewer.run()


if __name__ == "__main__":
    main()


