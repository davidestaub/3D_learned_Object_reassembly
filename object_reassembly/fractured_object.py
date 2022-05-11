import os
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Rotation as Rot, Translation, Pointcloud
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from itertools import permutations
import solver

from utils import nchoosek

np.random.seed(42)


class FracturedObject(object):

    def __init__(self, name):
        self.name = name
        self.fragments_orig = {}
        self.fragments = {}
        self.fragment_matches_gt = []
        self.fragment_matches = []
        self.kpts_orig = {}
        self.kpts = {}
        self.kpt_matches_gt = {}
        self.kpt_matches = {}
        self.transf_random = {}
        self.transf = {}

    # load fragment pointclouds and keypoints
    def load_object(self, path):

        new_path = path + self.name + "/cleaned/"

        print("Loading fragment meshes of object " + self.name + "...")

        for fragment in os.listdir(new_path):
            if fragment.endswith('.obj'):
                frag_no = int(fragment.rsplit(sep=".")[1])
                self.fragments_orig[frag_no] = Mesh.from_obj(filepath=new_path + fragment)
                self.fragments[frag_no] = Mesh.from_obj(filepath=new_path + fragment)

        print("Loading keypoints of object " + self.name + "...")

        new_path = path + self.name + "/processed/keypoints/"

        for kpts in os.listdir(new_path):
            frag_no = int(kpts.rsplit(sep=".")[1])
            self.kpts_orig[frag_no] = Pointcloud(np.load(new_path + kpts))
            self.kpts[frag_no] = Pointcloud(np.load(new_path + kpts))

    def save_object(self, path):
        # TODO: implement
        pass

    # load ground truth matches from file
    def load_gt(self, path, gt_from_closest=True):
        print("Loading ground truth matches of object " + self.name + "...")

        new_path = path + self.name + "/processed/matching/" + self.name + "_matching_matrix.npy"

        self.fragment_matches_gt = np.load(new_path).astype(bool)
        new_path = path + self.name + "/processed/matching/"

        if gt_from_closest:
            self.gt_from_closest()
        else:
            for matches in os.listdir(new_path):
                if matches.endswith(".npz"):
                    fragment0 = int(matches.rsplit(sep="_")[-2])
                    fragment1 = int(matches.rsplit(sep=".")[-2][-1])
                    self.kpt_matches_gt[(fragment0, fragment1)] = np.load(new_path + matches)

    # construct random transformation
    def create_random_pose(self):
        print("Creating random pose for object " + self.name + "...")
        for fragment in self.fragments.keys():
            # insert random rotations
            theta_x = np.random.uniform(0, 2 * np.pi)
            theta_y = np.random.uniform(0, 2 * np.pi)
            theta_z = np.random.uniform(0, 2 * np.pi)

            r = Rotation.from_euler(seq='xyz', angles=[theta_x, theta_y, theta_z])

            # insert random translation
            t = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

            self.transf_random[int(fragment)] = (r, t)

    def apply_random_transf(self):
        print("Applying random transformation to fragments...")
        for key, fragment in self.fragments.items():
            r_qut = self.transf_random[key][0].as_quat()
            Mesh.transform(fragment, Rot.from_quaternion(r_qut))
            Mesh.transform(fragment, Translation.from_vector(self.transf_random[key][1] + [0]))

        print("Applying random transformation to keypoints...")
        for key, points in self.kpts.items():
            r_qut = self.transf_random[key][0].as_quat()
            points.transform(Rot.from_quaternion(r_qut))
            points.transform(Translation.from_vector(self.transf_random[key][1] + [0]))

    def matching(self, use_gt=True, use_solver=True):
        fragments = range(len(self.fragments.keys()))
        combinations = list(permutations(fragments, 2))
        for idx, comb in enumerate(combinations):
            if (comb in self.kpt_matches_gt.keys() and self.kpt_matches_gt[comb] is not None) \
                    or (comb in self.kpt_matches.keys() and self.kpt_matches[comb] is not None):
                print("Combination " + str(idx) + "/" + str(len(combinations)) + " | Loop 1")
                print("Fragments (A, B):" + str(comb))

                # Get random poise(rp), ground truth(gt) keypoints of fragment A, B
                A_rand_pose_kpts = self.kpts[comb[0]]
                A_gt_pose_kpts = self.kpts_orig[comb[0]]
                B_rand_pose_kpts = self.kpts[comb[1]]
                B_gt_pose_kpts = self.kpts_orig[comb[1]]

                # Init corresponding keypoint pairs
                A_gt_pair = []
                A_rp_pair = []
                B_gt_pair = []
                B_rp_pair = []
                pair_nb = 0

                if use_gt:
                    if self.kpt_matches_gt[comb] is not None:
                        A_gt_idx,  B_gt_idx = self.kpt_matches_gt[comb]
                        for i in A_gt_idx:
                            A_gt_pair.append(self.kpts_orig[comb[0]].data['points'][i])
                            A_rp_pair.append(self.kpts[comb[0]].data['points'][i])
                        for i in B_gt_idx:
                            B_gt_pair.append(self.kpts_orig[comb[1]].data['points'][i])
                            B_rp_pair.append(self.kpts[comb[1]].data['points'][i])
                else:
                    if self.kpt_matches[comb] is not None:
                        A_gt_idx,  B_gt_idx = self.kpt_matches[comb]
                        for i in A_gt_idx:
                            A_gt_pair.append(self.kpts_orig[comb[0]].data['points'][i])
                            A_rp_pair.append(self.kpts[comb[0]].data['points'][i])
                        for i in B_gt_idx:
                            B_gt_pair.append(self.kpts_orig[comb[1]].data['points'][i])
                            B_rp_pair.append(self.kpts[comb[1]].data['points'][i])

                print("Number of keypoint pairs: "+str(len(A_rp_pair)))

                ptsA = np.array(A_rp_pair)
                ptsB = np.array(B_rp_pair)

                zcA = np.zeros((1, ptsA.shape[1]))
                ptsA_z = np.array([ptsA + zcA])
                zcB = np.zeros((1, ptsA.shape[1]))
                ptsB_z = np.array([ptsB + zcB])

                if use_solver:
                    solver.run_solver(ptsA_z, ptsB_z)


    # calculate ground truth from closest points
    def gt_from_closest(self, threshold=0.001):
        fragments = range(len(self.fragments.keys()))
        combinations = list(permutations(fragments, 2))

        for comb in combinations:
            if self.fragment_matches_gt[comb[0], comb[1]]:
                keypoints0 = np.array(self.kpts_orig[comb[0]].data['points'])
                keypoints1 = np.array(self.kpts_orig[comb[1]].data['points'])
                dists = cdist(keypoints0, keypoints1)
                close_enough_mask = np.min(dists, axis=0) < threshold
                closest = np.argmin(dists, axis=0)

                keypoint_assignment = np.zeros((keypoints0.shape[0], keypoints1.shape[0]))
                keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1
                if np.any(keypoint_assignment):
                    self.kpt_matches_gt[(comb[0], comb[1])] = np.nonzero(keypoint_assignment)
                else:
                    self.kpt_matches_gt[(comb[0], comb[1])] = None


    # def find_transformations_first3kpts(self):
    #     for fragment0 in range(len(self.fragments) - 1):
    #         for fragment1 in range(len(self.fragments) - 1):
    #             if self.fragment_matches[fragment0][fragment1] and (fragment0, fragment1) in self.kpt_matches.keys():
    #                 # pts0 = []
    #                 # for match in self.kpt_matches[(fragment0, fragment1)][0:3]:
    #                 #     pts0.append(self.fragments[fragment0].vertex[match])
    #                 #
    #                 # pts1 = []
    #                 # for match in self.kpt_matches[(fragment1, fragment0)][0:3]:
    #                 #     pts1.append(self.fragments[fragment1].vertex[match])
    #
    #                 pts0 = np.array(
    #                     [self.keypoints[fragment0][idx] for idx in self.kpt_matches[(fragment0, fragment1)][0:3]])
    #                 pts1 = np.array(
    #                     [self.keypoints[fragment1][idx] for idx in self.kpt_matches[(fragment1, fragment0)][0:3]])
    #
    #                 # centroid0 = np.mean(pts0, axis=1)
    #                 # centroid1 = np.mean(pts1, axis=1)
    #                 #
    #                 # centroid0 = centroid0.reshape(-1, 1)
    #                 # centroid1 = centroid1.reshape(-1, 1)
    #                 #
    #                 # m0 = pts0 - centroid0
    #                 # m1 = pts1 - centroid1
    #                 #
    #                 # H = m0 @ np.transpose(m1)
    #                 # U, S, Vt = np.linalg.svd(H)
    #                 # R = Vt.T @ U.T
    #                 #
    #                 # t = -R @ centroid0 + centroid1
    #
    #                 R, rmsd = Rotation.align_vectors(pts0, pts1)
    #
    #                 # insert random rotations for now:
    #                 # theta_x = np.random.uniform(0, 2*np.pi)
    #                 # theta_y = np.random.uniform(0, 2*np.pi)
    #                 # theta_z = np.random.uniform(0, 2*np.pi)
    #                 #
    #                 # r = Rotation.from_euler_angles([theta_x, theta_y, theta_z])
    #                 #
    #                 # R = r.data["matrix"]
    #
    #                 self.transformations[(fragment0, fragment1)] = R
