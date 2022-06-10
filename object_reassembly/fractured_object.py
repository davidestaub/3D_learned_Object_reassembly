import os
from collections import OrderedDict
from glob import glob
from itertools import permutations
from typing import Dict, List

import cvxpy as cp
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from cv2 import estimateAffine3D
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

import solver
from utils import helmert_nd

np.random.seed(42)


class FracturedObject(object):

    def __init__(self, path, graph_matching_method, keypoint_method='hybrid'):
        self.name = os.path.basename(path)
        self.path = path
        self.fragments_orig: Dict[int, Mesh] = {}
        self.fragments: Dict[int, Mesh] = {}
        self.fragment_matches_gt = []
        self.fragment_matches = []
        self.kpts_orig = {}
        self.kpts: Dict[int, Pointcloud] = {}
        self.kpt_matches_gt = {}
        self.kpt_matches = {}
        self.transf_random = {}  # key: N, value: (R,t), apply R(N)+t to move N from original position
        self.transf = OrderedDict()  # key: (A,B), value: (R,t), apply R(A)+t to match A to B
        self.constraints = None  # List of triplet constraints (a, b, c)
        self.N = -1
        self.keypoint_method = keypoint_method
        assert graph_matching_method in ['mst', 'original']
        self.graph_matching_method = graph_matching_method

    # load fragment pointclouds and keypoints
    def load_object(self):

        new_path = self.path + "/cleaned/"

        print("Loading fragment meshes of object " + self.name + "...")

        for fragment in os.listdir(new_path):
            if fragment.endswith('.obj'):
                frag_no = int(fragment.rsplit(sep=".")[1])
                self.fragments_orig[frag_no] = Mesh.from_obj(filepath=new_path + fragment)
                centroid = self.fragments_orig[frag_no].centroid()
                T = np.eye(4)
                T[0:3, 3] = centroid
                self.fragments_orig[frag_no].transform(np.linalg.inv(T))
                self.fragments[frag_no] = self.fragments_orig[frag_no].copy()

        self.N = len(self.fragments)
        print("Loading keypoints of object " + self.name + "...")

        keypoint_glob = os.path.join(self.path, "processed", "keypoints", f"keypoints_{self.keypoint_method}.*.npy")
        keypoint_files = glob(keypoint_glob)
        for i in range(keypoint_files):
            keypoint_path = os.path.join(self.path, "processed", "keypoints", f"keypoints_{self.keypoint_method}.{i}.npy")
            npy_kpts = np.load(keypoint_path)[:, 0:3]
            self.kpts_orig[i] = Pointcloud(npy_kpts)
            self.kpts[i] = Pointcloud(npy_kpts)

    def save_object(self, path):
        # TODO: implement
        pass

    # Load ground truth matches from file.
    def load_gt(self, gt_from_closest=False):
        print("Loading ground truth matches of object " + self.name + "...")

        new_path = self.path + "/processed/matching/" + self.name + "_matching_matrix.npy"
        print(new_path)

        self.fragment_matches_gt = np.load(new_path).astype(bool)

        keypoints_matchings_folder = os.path.join(self.path, 'predictions')
        
        if gt_from_closest:
            self.gt_from_closest()
        else:
            for matches in os.listdir(keypoints_matchings_folder):
                print("Loading matches ", matches)

                if matches.endswith(".npy"):
                    if "m0" in matches:
                        try:
                            fragment0 = int(matches.rsplit(sep="_")[-3])
                            fragment1 = int(matches.rsplit(sep="_")[-2])
                        except:
                            fragment0 = int(matches.rsplit(sep="_")[-4])
                            fragment1 = int(matches.rsplit(sep="_")[-3])

                        from_frag0_to_frag1 = np.load(os.path.join(keypoints_matchings_folder, matches))
                        from_frag0_to_frag1 = np.squeeze(from_frag0_to_frag1)

                        tuple_1 = []
                        tuple_2 = []

                        # Only take fragment pairs that have at least N points matched.
                        if (from_frag0_to_frag1 != -1).sum() >= 4:
                            for i in range(0, len(from_frag0_to_frag1)):
                                if from_frag0_to_frag1[i] != -1:
                                    tuple_1.append(i)
                                    tuple_2.append(from_frag0_to_frag1[i])

                            tuple_1 = np.array(tuple_1)
                            tuple_2 = np.array(tuple_2)
                            self.kpt_matches_gt[(fragment0, fragment1)] = (tuple_1, tuple_2)

    # construct random transformation
    def create_random_pose(self):
        rng = np.random.default_rng(seed=42)
        print("Creating random pose for object " + self.name + "...")
        for fragment in self.fragments.keys():
            # insert random rotations
            theta = rng.uniform(0, 2 * np.pi, size=(3,))
            R = Rotation.from_euler(seq='xyz', angles=theta).as_matrix()
            t = rng.uniform(-1, 1, size=(3,))
            self.transf_random[int(fragment)] = (R, t)

    def apply_transf(self, A, B):
        transf = self.transf[(A, B)]
        if transf[0] is not None and transf[1] is not None:
            print("Applying transformation from " + str(A) + " to " + str(B) + " to keypoints of fragment " + str(A))
            T = transform_from_rotm_tr(transf)
            self.kpts[A].transform(T)
            self.fragments[A].transform(T)

    def apply_random_transf(self):
        print("Applying random transformation to fragments...")
        for key, fragment in self.fragments.items():
            fragment.transform(transform_from_rotm_tr(self.transf_random[key]))

        print("Applying random transformation to keypoints...")
        for key, points in self.kpts.items():
            points.transform(transform_from_rotm_tr(self.transf_random[key]))

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

                elif find_t_method == "RANSAC_RIGID":

                    #minimum number of samples to get a transformation is 3 (per fragment) we use 4 because it helps get better transformations
                    min_number_pairs = 4

                    #Return NONE transformation if number of keypoint correspondences is not sufficient
                    if len(A_rp_pair) < min_number_pairs:
                        print("Not enough keypoint pairs, returning NAN transformation")
                        self.transf[comb] = (None, None)
                        nb_non_matches += 1
                    else:
                        # estimate with RANSAC
                        ransac = RansacEstimator(min_samples=3, residual_threshold=0.01, max_trials=1000)
                        ret = ransac.fit(Procrustes(), [ptsA, ptsB])
                        transform_ransac = ret["best_params"]
                        inliers_ransac = ret["best_inliers"]

                        #Get mean squared error of euclidean distance
                        mse_ransac = np.sqrt(Procrustes(transform_ransac).residuals(ptsA, ptsB).mean())
                        mse_ransac_inliers = np.sqrt(Procrustes(transform_ransac).residuals(ptsA[inliers_ransac], ptsB[inliers_ransac]).mean())
                        print("mse ransac all: {}".format(mse_ransac))
                        print("Number of ransac inliers: {}".format(sum(inliers_ransac)))
                        print("mse ransac inliers: {}".format(mse_ransac_inliers))

                        #Get final transformation matrix
                        R_mat = transform_ransac[:3, :3]
                        t = transform_ransac[:3, 3]
                        self.transf[comb] = (R_mat, t)
                        nb_non_matches += 1
                        match_pairwise[idx] = 1

                else:
                    # use gt for transformations
                    raise NotImplementedError

    def create_inverse_transformations_for_existing_pairs(self):
        """stores the transformatino from A -> B as the inverse transformation from B -> A.
                """
        dict_inv = {}
        for key in self.transf:
            A = key[0]
            B = key[1]
            current_transf = self.transf[key]
            T = transform_from_rotm_tr(current_transf)
            T_inv = np.linalg.inv(T)
            R_inv = T_inv[:3, :3]
            t_inv = T_inv[:3, 3]
            dict_inv[(B, A)] = (R_inv, t_inv)
        self.transf.update(dict_inv)

    def tripplet_matching(self, R_threshold, T_threshold):
        self.constraints = []
        fragments = range(len(self.fragments.keys()))
        comb_triplewise = list(permutations(fragments, 3))
        for i in range(0, len(comb_triplewise)):
            first = comb_triplewise[i][0]
            second = comb_triplewise[i][1]
            third = comb_triplewise[i][2]
            index_12 = (first, second)
            index_23 = (second, third)
            index_13 = (first, third)
            if index_12 in self.kpt_matches_gt and index_23 in self.kpt_matches_gt and index_13 in self.kpt_matches_gt:
                print("found potential triplet: ", index_12, index_23, index_13)

                T_12 = transform_from_rotm_tr(self.transf[index_12])
                T_31 = transform_from_rotm_tr(self.transf[(third, first)])
                T_32 = transform_from_rotm_tr(self.transf[(third, second)])

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
                    print(f"TRIPLET MATCH: {(first, second, third)}")
                else:
                    self.constraints += [(first, second, third)]
        print(self.constraints)

    def graph_matching_original(self):
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
        for i, z in enumerate(assignment):
            if z:
                a, b = idx_to_pair[i]
                vertices[a] += [b]
        return vertices

    def graph_matching_mst(self):
        # Minimum spanning tree with number of matching keypoints as weights.
        # Does not use triplet matching.
        edges = []
        for (a, b), (kpt_matches_a, kpt_matches_b) in self.kpt_matches_gt.items():
            edges.append((len(kpt_matches_a), a, b))
        edges = sorted(edges, reverse=True)
        cluster = list(range(self.N))
        vertices: List[List[int]] = [[] for _ in self.fragments]
        for _, a, b in edges:
            if cluster[a] == cluster[b]:
                continue
            # connect b to a
            vertices[a].append(b)
            vertices[b].append(a)
            cluster_b = cluster[b]
            for i in range(self.N):
                if cluster[i] == cluster_b:
                    cluster[i] = cluster[a]
        return vertices

    def find_final_transforms(self):
        if self.graph_matching_method == 'original':
            vertices = self.graph_matching_original()
        elif self.graph_matching_method == 'mst':
            vertices = self.graph_matching_mst()
        else:
            raise ValueError(f'No such graph matching method: {self.graph_matching_method}')

        for i, neighbors in enumerate(vertices):
            print(f"{i}: {neighbors}")

        # Traverse the graph to produce final transforms.
        visited = [False for _ in range(self.N)]
        self.final_transforms: Dict[int, (np.array, np.array)] = {}

        num_components = 0
        while not all(visited):
            # Start from the vertex that has the highest degree and has not been visited yet.
            x = max(range(len(visited)), key=lambda idx: len(vertices[idx]) if not visited[idx] else -1)

            # Render components far apart.
            component_translation = np.array([2, 0, 0]) * num_components
            queue = [(x, (np.eye(3), component_translation))]
            visited[x] = True
            while queue:
                x, transform = queue.pop(0)
                self.final_transforms[x] = transform

                for y in vertices[x]:
                    if not visited[y]:
                        y_transform = compose_transforms(transform, self.transf[y, x])
                        visited[y] = True
                        queue.append((y, y_transform))
            num_components += 1

        print(f"Matching graph resulted in {num_components} components.")
        for idx, fragment_mesh in enumerate(self.fragments):
            if idx in self.final_transforms:
                T = transform_from_rotm_tr(self.final_transforms[idx])
                self.kpts[idx].transform(T)
                self.fragments[idx].transform(T)
            else:
                print(f"Fragment {idx} has no final transformation.")

        # Reverse the initial random transformation of the starting fragment,
        # so the object stands neatly in the middle.
        starting_vertex = max(range(self.N), key=lambda idx: len(vertices[idx]))
        random_T = self.transf_random[starting_vertex]
        initial_random_transform = transform_from_rotm_tr(random_T)
        T = np.linalg.inv(initial_random_transform)
        for i in range(self.N):
            self.kpts[i].transform(T)
            self.fragments[i].transform(T)

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

        self._transform = transform_from_rotm_tr((rot, trans))

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


def transform_from_rotm_tr(transformation_pair):
    """Get The 4x4 Homogeneous Transformation matrix from the Rotation matrix and a translation vector.
            Args:
              transformation_pair: A pair containing the 3x3 Rotation matrix and a 1x3 translation vector
            Returns:
              A Homogeneous 4x4 Transformation matrix
            """
    rotation, translation = transformation_pair
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def compose_transforms(a, b):
    """Get the chained transformation from two transformations.
            Args:
              a: First Transformation
              b: Second Transformation
            Returns:
              The chained Transformatino TC = TA @ TB
            """
    Ta = transform_from_rotm_tr(a)
    Tb = transform_from_rotm_tr(b)

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
            best_params: The transformation found from the best model.
            inliers: a boolean mask indicating the inlier subset
              of the data for the best model.
        """
        best_inliers = None
        best_num_inliers = 0
        best_residual_sum = np.inf

        if not isinstance(data, (tuple, list)):
            data = [data]
        num_data, num_feats = data[0].shape

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
