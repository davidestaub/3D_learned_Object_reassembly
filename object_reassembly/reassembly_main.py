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


class FracturedObject(object):

    def __init__(self):
        self.fragments = []
        self.fragment_matches = []
        self.kpts = {}
        self.kpt_matches = {}
        self.transformations = {}

    def load_object(self, path):
        for object in os.listdir(path):
            if object.endswith('.obj'):
                self.fragments.append(Mesh.from_obj(path+object))

        matches_path = os.listdir(path+"matching/")[1]
        self.fragment_matches = np.load(path+"matching/"+matches_path)

        for idx, points in enumerate(os.listdir(path+"keypoints/")):
            self.kpts[idx] = np.load(path+"keypoints/"+points)

    def load_kpt_matches(self, matches0, matches1, idx0, idx1):
        self.kpt_matches[(idx0, idx1)] = matches0
        self.kpt_matches[(idx1, idx0)] = matches1

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

                    pad_row = np.array([[0, 0, 0, 1]])
                    pad_col = np.array([[0], [0], [0]])
                    tmp = np.concatenate((R, pad_col), axis=1)

                    R_pad = np.concatenate((tmp, pad_row))

                    # insert random rotations for now:
                    theta_x = np.random.uniform(0, 2*np.pi)
                    theta_y = np.random.uniform(0, 2*np.pi)
                    theta_z = np.random.uniform(0, 2*np.pi)

                    r = Rotation.from_euler_angles([theta_x, theta_y, theta_z])

                    R = r.data["matrix"]

                    self.transformations[(fragment0, fragment1)] = (R, t)


def main():

    dummy_matches0 = [0, 1, 2]
    dummy_matches1 = [0, 1, 2]

    bottle = FracturedObject()

    bottle.load_object(path)
    for i in range(10):
        for ii in range(10):
            if bottle.fragment_matches[i][ii]:
                bottle.load_kpt_matches(dummy_matches0, dummy_matches1, i, ii)
    bottle.find_transformations_first3kpts()

    viewer = App()
    viewer.add(bottle.fragments[0])
    viewer.show()

    # matched = bool(0)
    # for i in range(10):
    #     matched = 0
    #     ii = i
    #     while not matched:
    #         if bottle.fragment_matches[i][ii]:
    #             obj = viewer.add(bottle.fragments[i])
    #             R,t,T = bottle.transformations[(i, ii)]
    #             m = 0
    #             @viewer.on(interval=1000, frames=2)
    #             def move(f=100):
    #                 # obj.rotation = R
    #                 obj.translation = [f, 0, 0]
    #                 obj.update()
    #                 viewer.show()
    #             # Mesh.transform(bottle.fragments[i], T)
    #             # obj = viewer.add(bottle.fragments[i])
    #             matched = 1
    #         ii += 1
    #         if ii >= len(bottle.fragments):
    #             break

    # viewer.run()


if __name__ == "__main__":
    main()


