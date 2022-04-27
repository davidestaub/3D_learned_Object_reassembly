import os
from glob import glob

import numpy as np
import torch
import torch.utils.data as td
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


def pc_normalize(pc):
    """normalizes a pointcloud by centering and unit scaling"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def create_datasets(root, train_fraction, **dataset_kwargs):
    object_folders = [os.path.join(root, folder) for folder in os.listdir(root)]
    train_folders, test_folders, = train_test_split(
        object_folders,
        train_size=train_fraction,
        random_state=42
    )
    train_dataset = FragmentsDataset(train_folders, **dataset_kwargs)
    test_dataset = FragmentsDataset(test_folders, **dataset_kwargs)
    return train_dataset, test_dataset


# Creating a own dataset type for our input.
# dataset definition
class FragmentsDataset(td.Dataset):
    # load the dataset(
    def __init__(self, object_folders, normalize=True, overfit=False, match_with_inverted=False):
        self.dataset = []
        self.normalize = normalize
        self.overfit = overfit
        self.match_with_inverted = match_with_inverted

        # load the dataset
        for folder in object_folders:
            object_name = os.path.basename(folder)
            processed = os.path.join(folder, 'processed', '')
            matching = os.path.join(processed, 'matching', '')
            match_path = f'{matching}{object_name}_matching_matrix.npy'
            match_mat = np.load(match_path)

            # for each match pair load the keypoints, descripors and matches
            # also construct the gt assignment
            for i in range(match_mat.shape[0]):
                for j in range(i, match_mat.shape[0]):
                    # if the matching matrix is 1, two fragments should match
                    # extract all the keypoint information necessary
                    if match_mat[i, j] == 1:
                        item = {}
                        # original
                        if not self.overfit:
                            item['path_kpts_0'] = glob(os.path.join(processed, 'keypoints', f'*.{i}.npy'))[0]
                            item['path_kpts_1'] = glob(os.path.join(processed, 'keypoints', f'*.{j}.npy'))[0]
                            item['path_kpts_desc_0'] = \
                                glob(os.path.join(processed, 'keypoint_descriptors', f'*.{i}.npy'))[0]
                            item['path_kpts_desc_1'] = \
                                glob(os.path.join(processed, 'keypoint_descriptors', f'*.{j}.npy'))[0]
                            if self.match_with_inverted:
                                item['path_kpts_desc_inverted_0'] = \
                                    glob(os.path.join(processed, 'keypoint_descriptors_inverted', f'*.{i}.npy'))[0]
                                item['path_kpts_desc_inverted_1'] = \
                                    glob(os.path.join(processed, 'keypoint_descriptors_inverted', f'*.{j}.npy'))[0]
                            item['path_match_mat'] = glob(os.path.join(matching, f'*{j}_{i}.npz'))[0]
                        else:
                            item['path_kpts_0'] = glob(os.path.join(processed, 'keypoints', f'*.{i}.npy'))[0]
                            item['path_kpts_1'] = glob(os.path.join(processed, 'keypoints', f'*.{i}.npy'))[0]
                            item['path_kpts_desc_0'] = \
                                glob(os.path.join(processed, 'keypoint_descriptors', f'*.{i}.npy'))[0]
                            item['path_kpts_desc_1'] = \
                                glob(os.path.join(processed, 'keypoint_descriptors', f'*.{i}.npy'))[0]
                            if self.match_with_inverted:
                                item['path_kpts_desc_inverted_0'] = \
                                    glob(os.path.join(processed, 'keypoint_descriptors_inverted', f'*.{i}.npy'))[0]
                                item['path_kpts_desc_inverted_1'] = \
                                    glob(os.path.join(processed, 'keypoint_descriptors_inverted', f'*.{i}.npy'))[0]
                            item['path_match_mat'] = glob(os.path.join(matching, f'*{j}_{i}.npz'))[0]

                        self.dataset.append(item)

    # number of rows in the dataset
    def __len__(self):
        if self.match_with_inverted:
            return 2 * len(self.dataset)
        else:
            return len(self.dataset)

    # get a row at an index
    def __getitem__(self, idx):

        if self.match_with_inverted:
            # Every pair of fragments is fed to the network twice, once frag_0 and inverted frag_1 and once vice versa.
            inverted_0 = idx % 2 == 0
            idx = idx // 2
        # i is the keypoint index in the 0 cloud, item is the corresponding
        # cloud of potential matchings in the other fragment
        gtasg = np.array(load_npz(self.dataset[idx]['path_match_mat']).toarray(), dtype=np.float32)
        if not self.overfit:
            gt_matches0 = np.zeros(gtasg.shape[0]) - 1
            gt_matches1 = np.zeros(gtasg.shape[1]) - 1
            for i, kpts_j in enumerate(gtasg):
                for j, match in enumerate(kpts_j):
                    # if there is a match (1) then the keypoint in index i
                    # matches to the keypoint in index j of the other fragment
                    if match:
                        gt_matches0[i] = j
                        gt_matches1[j] = i
        else:
            gt_matches0 = np.array([i for i in range(gtasg.shape[0])])
            gt_matches1 = gt_matches0

        kp0_full = np.load(self.dataset[idx]['path_kpts_0'])
        kp1_full = np.load(self.dataset[idx]['path_kpts_1'])
        sc0 = kp0_full[:, -1]
        sc1 = kp1_full[:, -1]
        kp0 = kp0_full[:, :3]
        kp1 = kp1_full[:, :3]

        if self.normalize:
            kp0 = pc_normalize(kp0)
            kp1 = pc_normalize(kp1)

        # TODO, make pointnet in afterwards in forward pass from kpts and scores
        if self.match_with_inverted:
            if inverted_0:
                desc_path_0 = self.dataset[idx]['path_kpts_desc_inverted_0']
                desc_path_1 = self.dataset[idx]['path_kpts_desc_1']
            else:
                desc_path_0 = self.dataset[idx]['path_kpts_desc_0']
                desc_path_1 = self.dataset[idx]['path_kpts_desc_inverted_1']
        else:
            desc_path_0 = self.dataset[idx]['path_kpts_desc_0']
            desc_path_1 = self.dataset[idx]['path_kpts_desc_1']
        sample = {
            "keypoints0": torch.from_numpy(kp0.astype(np.float32)),
            "keypoints1": torch.from_numpy(kp1.astype(np.float32)),
            "scores0": torch.from_numpy(sc0.astype(np.float32)),
            "scores1": torch.from_numpy(sc1.astype(np.float32)),
            "descriptors0": torch.from_numpy(np.load(desc_path_0).astype(np.float32)),
            "descriptors1": torch.from_numpy(np.load(desc_path_1).astype(np.float32)),
            "gt_assignment": torch.from_numpy(gtasg),
            "gt_matches0": torch.from_numpy(gt_matches0),
            "gt_matches1": torch.from_numpy(gt_matches1)
        }

        return sample
