import os
from collections import OrderedDict
from itertools import permutations
from typing import Dict, List

import cvxpy as cp
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from compas.geometry import Rotation as Rot, Translation
from cv2 import estimateAffine3D
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

import solver
from utils import helmert_nd

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
        self.transf_random = {}  # key: N, value: (R,t), apply R(N)+t to move N from original position
        self.transf = OrderedDict()  # key: (A,B), value: (R,t), apply R(A)+t to match A to B
        self.constraints = None  # List of triplet constraints (a, b, c)

    # load fragment pointclouds and keypoints
    def load_object(self, path):

        new_path = path + self.name + "/cleaned/"

        print("Loading fragment meshes of object " + self.name + "...")

        for fragment in os.listdir(new_path):

            if fragment.endswith('.obj'):
                frag_no = int(fragment.rsplit(sep=".")[1])
                self.fragments_orig[frag_no] = Mesh.from_obj(filepath=new_path + fragment)
                self.fragments[frag_no] = Mesh.from_obj(filepath=new_path + fragment)
                centroid_orig = self.fragments_orig[frag_no].centroid()
                centroid = self.fragments[frag_no].centroid()
                T_orig = np.eye(4)
                T = np.eye(4)
                T[0:3,3] = centroid
                T_orig[0:3, 3] = centroid_orig
                self.fragments_orig[frag_no].transform(np.linalg.inv(T_orig))
                self.fragments[frag_no].transform(np.linalg.inv(T))


        print("Loading keypoints of object " + self.name + "...")

        new_path = path + self.name + "/processed/keypoints/"

        # for kpts in os.listdir(new_path):
        # frag_no = int(kpts.rsplit(sep=".")[1])
        # self.kpts_orig[frag_no] = Pointcloud(np.load(new_path + kpts))
        # self.kpts[frag_no] = Pointcloud(np.load(new_path + kpts))

        for kpts in os.listdir(new_path):
            frag_no = int(kpts.rsplit(sep=".")[1])
            print(frag_no)
            print(new_path + kpts)
            print(np.load(new_path + kpts))
            npy_kpts = np.load(new_path + kpts)[:, 0:3]
            print(npy_kpts)
            self.kpts_orig[frag_no] = Pointcloud(npy_kpts)
            self.kpts[frag_no] = Pointcloud(npy_kpts)

    def save_object(self, path):
        # TODO: implement
        pass

    # load ground truth matches from file
    def load_gt(self, path, gt_from_closest=False):
        print("Loading ground truth matches of object " + self.name + "...")

        new_path = path + self.name + "/processed/matching/" + self.name + "_matching_matrix.npy"

        self.fragment_matches_gt = np.load(new_path).astype(bool)
        new_path = path + self.name + "/processed/matching/"

        if gt_from_closest:
            self.gt_from_closest()
        else:
            for matches in os.listdir(new_path):
                print(new_path)

                if matches.endswith(".npz"):
                    fragment0 = int(matches.rsplit(sep="_")[-2])
                    fragment1 = int(matches.rsplit(sep=".")[-2][-1])
                    self.kpt_matches_gt[(fragment0, fragment1)] = np.load(new_path + matches)

                if matches.endswith(".npy"):
                    print("found npy")
                    # idk if this is good
                    if "m0" in matches:
                        fragment0 = int(matches.rsplit(sep="_")[-4])
                        fragment1 = int(matches.rsplit(sep="_")[-3])
                        from_frag0_to_frag1 = np.load(new_path + matches)

                        tuple_1 = []
                        tuple_2 = []
                        print(from_frag0_to_frag1)
                        for i in range(0, len(from_frag0_to_frag1)):
                            if from_frag0_to_frag1[i] != -1:
                                tuple_1.append(i)
                                tuple_2.append(from_frag0_to_frag1[i])

                        tuple_1 = np.array(tuple_1)
                        tuple_2 = np.array(tuple_2)
                        print(tuple_1)
                        print(tuple_2)
                        self.kpt_matches_gt[(fragment0, fragment1)] = (tuple_1, tuple_2)
                        print(fragment0, fragment1)
                        print(type(tuple_1))
                        print(type(tuple_2))
                        # self.kpt_matches_gt[(fragment0, fragment1)] = np.load(new_path + matches)

    # construct random transformation
    def create_random_pose(self):
        print("Creating random pose for object " + self.name + "...")
        for fragment in self.fragments.keys():
            # insert random rotations
            theta_x = np.random.uniform(0, 2 * np.pi)
            theta_y = np.random.uniform(0, 2 * np.pi)
            theta_z = np.random.uniform(0, 2 * np.pi)

            R = Rotation.from_euler(seq='xyz', angles=[theta_x, theta_y, theta_z])

            # insert random translation
            t = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

            self.transf_random[int(fragment)] = (R, t)

    def apply_transf(self, A, B):
        if self.transf[(A, B)][0] is not None and self.transf[(A, B)][1] is not None:
            # r_qut = self.transf[(A, B)][0].as_quat()
            # print("Applying transformation from " + str(A) + " to " + str(B) + " to fragment " + str(A))
            # Mesh.transform(self.fragments[A], Rot.from_quaternion(r_qut))
            # Mesh.transform(self.fragments[A], Translation.from_vector(self.transf[(A, B)][1] + [0]))

            print("Applying transformation from " + str(A) + " to " + str(B) + " to keypoints of fragment " + str(A))

            T = transform_from_rotm_tr(self.transf[(A, B)][0], self.transf[(A, B)][1])
            self.kpts[A].transform(T)
            Mesh.transform(self.fragments[A], T)

            # self.kpts[A].transform(Rot.from_quaternion(r_qut))
            # self.kpts[A].transform(Translation.from_vector(self.transf[(A, B)][1] + [0]))

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

    def find_transformations(self, use_gt=True, find_t_method="RANSAC_RIGID", s_min=0.1, use_rigid_transform=True):
        fragments = range(len(self.fragments.keys()))  # nof fragments
        combinations = list(permutations(fragments, 2))  # all possible fragment pairs
        nb_non_matches = 0

        match_pairwise = np.zeros((len(combinations), 1))  # pairwise matching array

        for idx, comb in enumerate(combinations):
            if (comb in self.kpt_matches_gt.keys() and self.kpt_matches_gt[comb] is not None) \
                    or (comb in self.kpt_matches.keys() and self.kpt_matches[comb] is not None):
                print("Combination " + str(idx) + "/" + str(len(combinations)) + " | Loop 1")
                print("Fragments (A, B):" + str(comb))

                # Get random pose(rp), ground truth(gt) keypoints of fragment A, B
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
                        print(comb)
                        print(self.kpt_matches_gt[comb])
                        A_gt_idx, B_gt_idx = self.kpt_matches_gt[comb]
                        for i in A_gt_idx:
                            A_gt_pair.append(self.kpts_orig[comb[0]].data['points'][i])
                            A_rp_pair.append(self.kpts[comb[0]].data['points'][i])
                        for i in B_gt_idx:
                            B_gt_pair.append(self.kpts_orig[comb[1]].data['points'][i])
                            B_rp_pair.append(self.kpts[comb[1]].data['points'][i])
                else:
                    if self.kpt_matches[comb] is not None:
                        A_gt_idx, B_gt_idx = self.kpt_matches[comb]
                        for i in A_gt_idx:
                            A_gt_pair.append(self.kpts_orig[comb[0]].data['points'][i])
                            A_rp_pair.append(self.kpts[comb[0]].data['points'][i])
                        for i in B_gt_idx:
                            B_gt_pair.append(self.kpts_orig[comb[1]].data['points'][i])
                            B_rp_pair.append(self.kpts[comb[1]].data['points'][i])

                print("Number of keypoint pairs: " + str(len(A_rp_pair)))

                ptsA = np.array(A_rp_pair)
                ptsB = np.array(B_rp_pair)

                zcA = np.zeros((1, ptsA.shape[1]))
                ptsA_z = np.array([ptsA + zcA])
                zcB = np.zeros((1, ptsA.shape[1]))
                ptsB_z = np.array([ptsB + zcB])

                if find_t_method == "solver":
                    sol = solver.run_solver(ptsA_z, ptsB_z)
                    if sol is not None:
                        if sol["s_opt"] > s_min and sum(abs(sol["t"])) > 1e-6:
                            print("Valid solution for T, s = " + str(sol["s_opt"]))
                            R_mat = sol["R"]
                            t = sol["t"]
                            if use_rigid_transform:
                                R_mat, c, t = helmert_nd(ptsA, ptsB, sol["s_opt"], sol["R"], sol["t"])
                            R = Rotation.from_matrix(R_mat)
                            self.transf[comb] = (R, t)
                            nb_non_matches += 1
                            match_pairwise[idx] = 1

                    else:
                        print("No valid solution for T -> creating NaN R, T")
                        self.transf[comb] = (None, None)
                        nb_non_matches += 1
                elif find_t_method == "RANSAC":
                    retval, out, inliers = estimateAffine3D(ptsA, ptsB, confidence=0.99)
                    if not retval:
                        print("Transformation estimation unsuccessful -> creating NaN R, T")
                        self.transf[comb] = (None, None)
                        nb_non_matches += 1
                    else:
                        print("Valid T estimated, " + str(np.sum(inliers)) + "/" + str(len(A_rp_pair)) + " inliers")
                        R_mat = out[:, :3]
                        t = out[:, 3]
                        print(out)
                        R = Rotation.from_matrix(R_mat)
                        self.transf[comb] = (R, t)
                        nb_non_matches += 1
                        match_pairwise[idx] = 1

                elif find_t_method == "RANSAC_RIGID":
                    min_number_pairs = 4
                    if len(A_rp_pair) < min_number_pairs:
                        print("Not enough keypoint pairs, returning NAN transformation")
                        self.transf[comb] = (None, None)
                        nb_non_matches += 1
                    else:
                        naive_model = Procrustes()
                        naive_model.estimate(ptsA, ptsB)
                        transform_naive = naive_model.params
                        mse_naive = np.sqrt(naive_model.residuals(ptsA, ptsB).mean())
                        print("mse naive: {}".format(mse_naive))

                        # estimate with RANSAC
                        ransac = RansacEstimator(min_samples=3, residual_threshold=(0.01), max_trials=100, )
                        ret = ransac.fit(Procrustes(), [ptsA, ptsB])
                        transform_ransac = ret["best_params"]
                        inliers_ransac = ret["best_inliers"]
                        mse_ransac = np.sqrt(Procrustes(transform_ransac).residuals(ptsA, ptsB).mean())
                        print("mse ransac all: {}".format(mse_ransac))
                        mse_ransac_inliers = np.sqrt(
                            Procrustes(transform_ransac).residuals(ptsA[inliers_ransac], ptsB[inliers_ransac]).mean())
                        print("mse ransac inliers: {}".format(mse_ransac_inliers))

                        R_mat = transform_ransac[:3, :3]
                        t = transform_ransac[:3, 3]
                        self.transf[comb] = (R_mat, t)
                        nb_non_matches += 1
                        match_pairwise[idx] = 1

                else:
                    # use gt for transformations
                    raise NotImplementedError

    def create_inverse_transformations_for_existing_pairs(self):
        # For now I am just takinh the inverses, later we could also use m1 and notjust m0 files and compute these transformatinos like they do
        # in the matlab script
        dict_inv = {}
        for key in self.transf:
            A = key[0]
            B = key[1]
            current_transf = self.transf[key]
            T = transform_from_rotm_tr(current_transf[0], current_transf[1])
            T_inv = np.linalg.inv(T)
            R_inv = T_inv[:3, :3]
            t_inv = T_inv[:3, 3]
            dict_inv[(B, A)] = (R_inv, t_inv)
        self.transf.update(dict_inv)

    def tripplet_matching(self, R_threshold, T_threshold):
        self.constraints = []
        fragments = range(len(self.fragments.keys()))
        comb_triplewise = list(permutations(fragments, 3))
        print(comb_triplewise)
        for i in range(0, len(comb_triplewise)):
            first = comb_triplewise[i][0]
            second = comb_triplewise[i][1]
            third = comb_triplewise[i][2]
            print(first, second, third)
            index_12 = (first, second)
            index_23 = (second, third)
            index_13 = (first, third)
            if index_12 in self.kpt_matches_gt and index_23 in self.kpt_matches_gt and index_13 in self.kpt_matches_gt:
                print("found potential triplet: ", index_12, index_23, index_13)

                T_12 = transform_from_rotm_tr(self.transf[index_12][0], self.transf[index_12][1])
                T_31 = transform_from_rotm_tr(self.transf[(third, first)][0], self.transf[(third, first)][1])
                T_32 = transform_from_rotm_tr(self.transf[(third, second)][0], self.transf[(third, second)][1])
                print(T_32, T_31, T_12)

                T_32_est = T_12 @ T_31

                R_32 = T_32[0:3, 0:3]
                R_32_est = T_32_est[0:3, 0:3]
                angle_R_32 = np.arccos((np.trace(R_32) - 1) * 0.5)
                angle_R_32_est = np.arccos((np.trace(R_32_est) - 1) * 0.5)
                angle_diff = np.abs(angle_R_32 - angle_R_32_est)

                Transl_32 = T_32[0:3, 3]
                Transl_32_est = T_32_est[0:3, 3]
                distance_diff = np.linalg.norm(Transl_32 - Transl_32_est)

                if distance_diff <= T_threshold and angle_diff <= R_threshold:
                    self.constraints += [(first, second, third)]
                    print(f"TRIPLET MATCH: {(first, second, third)}")
                else:
                    print("NO Match")
        print(self.constraints)

    def find_final_transforms(self):
        assert self.constraints is not None, f"Perform triplet matching."

        # Solve optimization problem.
        idx_to_pair = [(a, b) for a, b in self.transf]
        pair_to_idx = {p: i for i, p in enumerate(idx_to_pair)}

        zs = cp.Variable(len(self.transf), boolean=True)
        objective = cp.sum(zs)
        # Triplet constraint.
        constraints = [zs[pair_to_idx[a, b]] + zs[pair_to_idx[b, c]] + zs[pair_to_idx[c, a]] <= 2 for a, b, c in
                       self.constraints]
        # Symmetry constraint.
        constraints += [zs[pair_to_idx[a, b]] == zs[pair_to_idx[b, a]] for a, b in pair_to_idx if a < b]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()
        assignment = problem.solution.primal_vars[1]

        vertices: List[List[int]] = [[] for _ in self.fragments]
        degree = {}
        for i, z in enumerate(assignment):
            if z:
                a, b = idx_to_pair[i]
                vertices[a] += [b]
                degree[a] = degree.get(a, 0) + 1

        for i, neighbors in enumerate(vertices):
            print(f"{i}: {neighbors}")

        x = max(degree, key=degree.get)

        queue = [(x,  (np.eye(3), (np.zeros(3))))]
        visited = [False for _ in self.fragments]
        visited[x] = True
        self.final_transforms: Dict[int, (np.array, np.array)] = {}

        while queue:
            x, transform = queue.pop(0)
            self.final_transforms[x] = transform

            for y in vertices[x]:
                if not visited[y]:
                    y_transform = compose_transforms(transform, self.transf[y, x])
                    visited[y] = True
                    queue.append((y, y_transform))

        for idx, fragment_mesh in enumerate(self.fragments):
            T = transform_from_rotm_tr(*self.final_transforms[idx])
            self.kpts[idx].transform(T)
            Mesh.transform(self.fragments[idx], T)



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


class Procrustes:
    """Determines the best rigid transform [1] between two point clouds.
    References:
      [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """

    def __init__(self, transform=None):
        self._transform = transform

    def __call__(self, xyz):
        return Procrustes.transform_xyz(xyz, self._transform)

    @staticmethod
    def transform_xyz(xyz, transform):
        """Applies a rigid transform to an (N, 3) point cloud.
        """
        xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
        xyz_t_h = (transform @ xyz_h.T).T  # apply transform
        return xyz_t_h[:, :3]

    def estimate(self, X, Y):
        # find centroids
        X_c = np.mean(X, axis=0)
        Y_c = np.mean(Y, axis=0)

        # shift
        X_s = X - X_c
        Y_s = Y - Y_c

        # compute SVD of covariance matrix
        cov = Y_s.T @ X_s
        u, _, vt = np.linalg.svd(cov)

        # determine rotation
        rot = u @ vt
        if np.linalg.det(rot) < 0.:
            vt[2, :] *= -1
            rot = u @ vt

        # determine optimal translation
        trans = Y_c - rot @ X_c

        self._transform = transform_from_rotm_tr(rot, trans)

    def residuals(self, X, Y):
        """L2 distance between point correspondences.
        """
        # print("X=",X)
        Y_est = self(X)
        # print("Y_est=",Y_est)
        # print("Y=",Y)
        sum_sq = np.sum((Y_est - Y) ** 2, axis=1)
        return sum_sq

    @property
    def params(self):
        return self._transform


def transform_from_rotm_tr(rotm, tr):
    transform = np.eye(4)
    transform[:3, :3] = rotm
    transform[:3, 3] = tr
    return transform

def compose_transforms(a, b):
    Ta = transform_from_rotm_tr(*a)
    Tb = transform_from_rotm_tr(*b)

    Tc = Ta @ Tb
    return Tc[:3, :3], Tc[:3, 3]


class RansacEstimator:
    """Random Sample Consensus.
    """

    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100):
        """Constructor.
        Args:
          min_samples: The minimal number of samples needed to fit the model
            to the data. If `None`, we assume a linear model in which case
            the minimum number is one more than the feature dimension.
          residual_threshold: The maximum allowed residual for a sample to
            be classified as an inlier. If `None`, the threshold is chosen
            to be the median absolute deviation of the target variable.
          max_trials: The maximum number of trials to run RANSAC for. By
            default, this value is 100.
        """
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials

    def fit(self, model, data):
        """Robustely fit a model to the data.
        Args:
          model: a class object that implements `estimate` and
            `residuals` methods.
          data: the data to fit the model to. Can be a list of
            data pairs, such as `X` and `y` in the case of
            regression.
        Returns:
          A dictionary containing:
            best_model: the model with the largest consensus set
              and lowest residual error.
            inliers: a boolean mask indicating the inlier subset
              of the data for the best model.
        """
        best_model = None
        best_inliers = None
        best_num_inliers = 0
        best_residual_sum = np.inf

        if not isinstance(data, (tuple, list)):
            data = [data]
        num_data, num_feats = data[0].shape

        print(num_feats)

        if self.min_samples is None:
            self.min_samples = num_feats + 1
        if self.residual_threshold is None:
            if len(data) > 1:
                data_idx = 1
            else:
                data_idx = 0
            self.residual_threshold = np.median(np.abs(
                data[data_idx] - np.median(data[data_idx])))

        for trial in range(self.max_trials):
            # randomly select subset
            rand_subset_idxs = np.random.choice(np.arange(num_data), size=self.min_samples, replace=False)
            rand_subset = [d[rand_subset_idxs] for d in data]

            # estimate with model
            model.estimate(*rand_subset)

            # compute residuals
            residuals = model.residuals(*data)
            residuals_sum = residuals.sum()
            inliers = residuals <= self.residual_threshold
            num_inliers = np.sum(inliers)

            # decide if better
            if (best_num_inliers < num_inliers) or (best_residual_sum > residuals_sum):
                best_num_inliers = num_inliers
                best_residual_sum = residuals_sum
                best_inliers = inliers

        # refit model using all inliers for this set
        if best_num_inliers == 0:
            data_inliers = data
        else:
            data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)

        ret = {
            "best_params": model.params,
            "best_inliers": best_inliers,
        }
        return ret


def add_to_viewer(elements: list, viewer):
    for element in elements:
        viewer.add(element)
