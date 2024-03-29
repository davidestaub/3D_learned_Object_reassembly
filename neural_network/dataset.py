import os
from glob import glob
import numpy as np
import torch
import torch.utils.data as td
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from neural_network.utils.conf import *


def pc_normalize(pc):
    """
    It takes a point cloud as input and returns a point cloud with the same shape, but with the
    centroid at the origin
    
    :param pc: The point cloud to be normalized
    :return: the point cloud after it has been normalized.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc


def create_datasets(root, conf):
    """
    It takes a root folder, and splits it into train and test folders
    
    :param root: the path to the folder containing the object folders
    :param conf: a dictionary containing the following keys:
    :return: A tuple of two datasets, one for training and one for testing.
    """
    object_folders = [os.path.join(root, folder) for folder in os.listdir(root)]

    train_folders, test_folders, = train_test_split(
        object_folders,
        train_size = conf['train_fraction'],
        random_state = conf['seed']
    )
    train_dataset = DatasetTrain(train_folders, conf)
    test_dataset = DatasetTrain(test_folders, conf)
    return train_dataset, test_dataset


class DatasetPredict(td.Dataset):
    def __init__(self, folder_root, conf, single_object=False):
        # Set single object to true if `folder_root` points directly to the object directory. Set it to False if it's a
        # folder containing multiple object folders.
        self.dataset = []
        self.pillar = conf['pillar']
        self.normalize = conf['normalize_data']
        self.overfit = conf['overfit']
        self.match_with_inverted = conf['match_inverted']
        # setup for paths
        if single_object:
            object_folders = [folder_root]
        else:
            object_folders = [os.path.join(folder_root, folder) for folder in os.listdir(folder_root) if not "prediction" in folder]
        
        kpt_desc = 'keypoint_descriptors'
        kpt_desc_inv = 'keypoint_descriptors_inverted'
        kpts_method = conf['kpts']
        desc_method = '_'.join([conf['kpts'], conf['desc']])

        # correct settings of hyperpillar
        if conf['pillar']:
            #kpt_desc = '_'.join(['pillar', kpt_desc])
            #kpt_desc_inv = '_'.join(['pillar', kpt_desc_inv])
            self.match_with_inverted = False
            desc_method = '_'.join([conf['kpts'], 'pillar'])
        
        for folder in object_folders:
            processed = os.path.join(folder, 'processed')
            kpts_path = os.path.join(processed, 'keypoints')

            num_files = len(glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.*.npy')))

            for i in range(num_files):
                for j in range(i+1, num_files):
                    item = {}
                    object_name = os.path.basename(folder)
                    item['pairname'] = '_'.join([folder, str(i), str(j)])
                    try:
                        item['path_kpts_0'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{i}.npy'))[0]
                        item['path_kpts_1'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{j}.npy'))[0]
                        item['path_kpts_desc_0'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{i}.npy'))[0]
                        item['path_kpts_desc_1'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{j}.npy'))[0]
                        if self.match_with_inverted:
                            item['path_kpts_desc_inverted_0'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{i}.npy'))[0]
                            item['path_kpts_desc_inverted_1'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{j}.npy'))[0]
                    except Exception as e:
                        print(f"WARNING: Error loading objects in folder {folder}: {e}")
                        continue
            
                    self.dataset.append(item)


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

        kp0_full = np.load(self.dataset[idx]['path_kpts_0'])
        kp1_full = np.load(self.dataset[idx]['path_kpts_1'])
        sc0 = kp0_full[:, 3]
        sc1 = kp1_full[:, 3]
        kp0 = kp0_full[:, :3]
        kp1 = kp1_full[:, :3]

        if self.normalize:
            kp0 = pc_normalize(kp0)
            kp1 = pc_normalize(kp1)

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
        # load descriptors
        des0 = np.load(desc_path_0)
        des1 = np.load(desc_path_1)
        #zero pad if needed
        if not self.pillar:
            diff =  model_conf['descriptor_dim'] - des0.shape[1]
            if diff < 0:
                exit("ERROR: FEATURES ARE BIGGER THAN STATED FEATURE DIMENSION!")
            if diff > 0:
                des0 = np.concatenate((des0, np.zeros((des0.shape[0], diff))), axis=1)
                des1 = np.concatenate((des1, np.zeros((des1.shape[0], diff))), axis=1)

        sample = {
            "keypoints0": torch.from_numpy(kp0.astype(np.float32)).unsqueeze(0),
            "keypoints1": torch.from_numpy(kp1.astype(np.float32)).unsqueeze(0),
            "scores0": torch.from_numpy(sc0.astype(np.float32)).unsqueeze(0),
            "scores1": torch.from_numpy(sc1.astype(np.float32)).unsqueeze(0),
            "descriptors0": torch.from_numpy(des0.astype(np.float32)).unsqueeze(0),
            "descriptors1": torch.from_numpy(des1.astype(np.float32)).unsqueeze(0),
            "pair_name": self.dataset[idx]['pairname']
        }

        return sample


# Creating a own dataset type for our input.
# dataset definition
class DatasetTrain(td.Dataset):
    # load the dataset(
    def __init__(self, object_folders, conf):
        self.dataset = []
        self.pillar = conf['pillar']
        self.normalize = conf['normalize_data']
        self.overfit = conf['overfit']
        self.match_with_inverted = conf['match_inverted']

        kpt_desc = 'keypoint_descriptors'
        kpt_desc_inv = 'keypoint_descriptors_inverted'
        kpts_method = conf['kpts']
        desc_method = '_'.join([conf['kpts'], conf['desc']])

        # correct settings of hyperpillar
        if conf['pillar']:
            #kpt_desc = '_'.join(['pillar', kpt_desc])
            #kpt_desc_inv = '_'.join(['pillar', kpt_desc_inv])
            self.match_with_inverted = False
            desc_method = '_'.join([conf['kpts'], 'pillar'])

        # load the dataset
        for folder in object_folders:

            object_name = os.path.basename(folder)
            processed = os.path.join(folder, 'processed')
            kpts_path = os.path.join(processed, 'keypoints')
            matching = os.path.join(processed, 'matching')
            match_path = os.path.join(matching, f'{object_name}_matching_matrix.npy')
            match_mat = np.load(match_path)
            num_files = len(glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.*.npy')))

            # for each match pair load the keypoints, descripors and matches
            # also construct the gt assignment

            for i in range(num_files):
                for j in range(num_files):
                    # if the matching matrix is 1, two fragments should match
                    # extract all the keypoint information necessary
                    if match_mat[i, j] == 1:
                        item = {}
                        # original
                        try:
                            if not self.overfit:
                                item['pairname'] = '_'.join([folder, str(i), str(j)])
                                item['path_kpts_0'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{i}.npy'))[0]
                                item['path_kpts_1'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{j}.npy'))[0]
                                item['path_kpts_desc_0'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{i}.npy'))[0]
                                item['path_kpts_desc_1'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{j}.npy'))[0]
                                if self.match_with_inverted:
                                    item['path_kpts_desc_inverted_0'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{i}.npy'))[0]
                                    item['path_kpts_desc_inverted_1'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{j}.npy'))[0]
                                item['path_match_mat'] = glob(os.path.join(matching, f'*{desc_method}_{max(i,j)}_{min(i,j)}.npz'))[0]
                            else:
                                item['path_kpts_0'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{i}.npy'))[0]
                                item['path_kpts_1'] = glob(os.path.join(kpts_path, f'keypoints_{kpts_method}.{i}.npy'))[0]
                                item['path_kpts_desc_0'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{i}.npy'))[0]
                                item['path_kpts_desc_1'] = glob(os.path.join(processed, kpt_desc, f'*{desc_method}.{i}.npy'))[0]
                                if self.match_with_inverted:
                                    item['path_kpts_desc_inverted_0'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{i}.npy'))[0]
                                    item['path_kpts_desc_inverted_1'] = glob(os.path.join(processed, kpt_desc_inv, f'*{desc_method}.{i}.npy'))[0]
                                item['path_match_mat'] = glob(os.path.join(matching, f'*{desc_method}_{max(i,j)}_{min(i,j)}.npz'))[0]
                        except Exception as e:
                            print(f"Error loading objects in folder {folder}: {e}")
                            exit(0)
                    
                        # throw out samples where we have too little matches
                        # since we have two same matched (i,j) (j,i) divide by 2
                        gt = np.array(load_npz(item['path_match_mat']).toarray(), dtype=np.float32)
                        num_matches = np.sum(gt) / 2
                        num = gt.shape[0]
                        if num_matches / num < conf['gt_match_thresh']:
                            continue

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
        sc0 = kp0_full[:, 3]
        sc1 = kp1_full[:, 3]
        kp0 = kp0_full[:, :3]
        kp1 = kp1_full[:, :3]

        if self.normalize:
            kp0 = pc_normalize(kp0)
            kp1 = pc_normalize(kp1)

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
        # load descriptors
        des0 = np.load(desc_path_0)
        des1 = np.load(desc_path_1)

        #zero pad if needed
        if not self.pillar:
            diff =  model_conf['descriptor_dim'] - des0.shape[1]
            if diff < 0:
                exit("ERROR: FEATURES ARE BIGGER THAN STATED FEATURE DIMENSION!")
            if diff > 0:
                des0 = np.concatenate((des0, np.zeros((des0.shape[0], diff))), axis=1)
                des1 = np.concatenate((des1, np.zeros((des1.shape[0], diff))), axis=1)

        sample = {
            "keypoints0": torch.from_numpy(kp0.astype(np.float32)),
            "keypoints1": torch.from_numpy(kp1.astype(np.float32)),
            "scores0": torch.from_numpy(sc0.astype(np.float32)),
            "scores1": torch.from_numpy(sc1.astype(np.float32)),
            "descriptors0": torch.from_numpy(des0.astype(np.float32)),
            "descriptors1": torch.from_numpy(des1.astype(np.float32)),
            "gt_assignment": torch.from_numpy(gtasg),
            "gt_matches0": torch.from_numpy(gt_matches0),
            "gt_matches1": torch.from_numpy(gt_matches1),
            "pair_name": self.dataset[idx]['pairname']
        }

        return sample
