import os
import numpy as np
# from compas.datastructures import Mesh
# from compas.geometry import Rotation
from scipy.spatial.transform import Rotation

np.random.seed(42)


class FracturedObject(object):

    def __init__(self, name):
        self.name = name
        self.fragments = {}
        self.fragment_matches_gt = []
        self.fragment_matches = []
        self.kpts = {}
        self.kpt_matches_gt = {}
        self.kp_matches = {}
        self.random_transf = {}
        self.transformations = {}

    # load fragment pointclouds and keypoints
    def load_object(self, path):

        print("Loading fragments of object " + self.name + "...")

        new_path = path + self.name + "/cleaned/"

        for fragment in os.listdir(new_path):
            if fragment.endswith('.npy'):
                frag_no = fragment.rsplit(sep=".")[1]
                self.fragments[frag_no]=np.load(new_path+fragment)

        print("Loading keypoints of object " + self.name + "...")

        new_path = path + self.name + "/processed/keypoints/"

        for kpts in os.listdir(new_path):
            frag_no = kpts.rsplit(sep=".")[1]
            self.kpts[frag_no] = np.load(new_path + kpts)

    def load_gt(self, path):

        print("Loading ground truth matches of object " + self.name + "...")

        new_path = path + self.name + "/processed/matching/" + self.name + "_matching_matrix.npy"

        self.fragment_matches_gt = np.load(new_path)
        new_path = path + self.name + "/processed/matching/"

        for matches in os.listdir(new_path):
            if matches.endswith(".npz"):
                fragment0 = int(matches.rsplit(sep="_")[-2])
                fragment1 = int(matches.rsplit(sep=".")[-2][-1])
                self.kpt_matches_gt[(fragment0, fragment1)] = np.load(new_path+matches)

    def load_kpt_matches(self, matches0, matches1, idx0, idx1):
        self.kpt_matches[(idx0, idx1)] = matches0
        self.kpt_matches[(idx1, idx0)] = matches1

    def create_random_pose(self):
        for fragment in self.fragments.keys():

            # insert random rotations
            theta_x = np.random.uniform(0, 2*np.pi)
            theta_y = np.random.uniform(0, 2*np.pi)
            theta_z = np.random.uniform(0, 2*np.pi)

            r = Rotation.from_euler(seq='xyz', angles=[theta_x, theta_y, theta_z])

            # insert random translation
            t = [np.random.uniform(), np.random.uniform(), np.random.uniform()]

            self.random_transf[fragment] = (r.as_matrix(), t)

    def plot_kpts_ptclouds(self):
        # TODO: implement
        pass

    def find_transformations_first3kpts(self):
        for fragment0 in range(len(self.fragments) - 1):
            for fragment1 in range(len(self.fragments) - 1):
                if self.fragment_matches[fragment0][fragment1] and (fragment0, fragment1) in self.kpt_matches.keys():
                    # pts0 = []
                    # for match in self.kpt_matches[(fragment0, fragment1)][0:3]:
                    #     pts0.append(self.fragments[fragment0].vertex[match])
                    #
                    # pts1 = []
                    # for match in self.kpt_matches[(fragment1, fragment0)][0:3]:
                    #     pts1.append(self.fragments[fragment1].vertex[match])

                    pts0 = np.array(
                        [self.keypoints[fragment0][idx] for idx in self.kpt_matches[(fragment0, fragment1)][0:3]])
                    pts1 = np.array(
                        [self.keypoints[fragment1][idx] for idx in self.kpt_matches[(fragment1, fragment0)][0:3]])

                    # centroid0 = np.mean(pts0, axis=1)
                    # centroid1 = np.mean(pts1, axis=1)
                    #
                    # centroid0 = centroid0.reshape(-1, 1)
                    # centroid1 = centroid1.reshape(-1, 1)
                    #
                    # m0 = pts0 - centroid0
                    # m1 = pts1 - centroid1
                    #
                    # H = m0 @ np.transpose(m1)
                    # U, S, Vt = np.linalg.svd(H)
                    # R = Vt.T @ U.T
                    #
                    # t = -R @ centroid0 + centroid1

                    R, rmsd = Rotation.align_vectors(pts0, pts1)

                    # insert random rotations for now:
                    # theta_x = np.random.uniform(0, 2*np.pi)
                    # theta_y = np.random.uniform(0, 2*np.pi)
                    # theta_z = np.random.uniform(0, 2*np.pi)
                    #
                    # r = Rotation.from_euler_angles([theta_x, theta_y, theta_z])
                    #
                    # R = r.data["matrix"]

                    self.transformations[(fragment0, fragment1)] = R
