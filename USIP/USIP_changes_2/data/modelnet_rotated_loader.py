### ulsteger: all changes made are marked with "ulsteger"

import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import sys

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py

# matvogel: set the USIP directory to sys path
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.chdir('..')
usip_root = os.getcwd()
sys.path.insert(0,usip_root)

import modelnet.config as config



def make_dataset_modelnet40(root):
    # matvogel: easier load of whole input folder
    #           also added the filename to not just deal with numbers
    dataset = []
    #idx = 0
    for item in os.listdir(root):
        if item.split('.')[-1] == ("npy"):
            dataset.append((os.path.join(root,item), item, 0))
        else:
            os.remove(os.path.join(root,item))
        #idx += 1
    return dataset


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class ModelNet_Rotated_Loader(data.Dataset):
    def __init__(self, root, opt):
        super(ModelNet_Rotated_Loader, self).__init__()
        self.root = root
        print("Data Loader initialized with root", root)
        self.opt = opt

        self.dataset = make_dataset_modelnet40(self.root)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        ### ulsteger: added options for scaling and too few vertices
        pc_np_file, pc_filename, pc_type_id = self.dataset[index]

        data = np.load(pc_np_file)
        
        
        ### ulsteger: UNCOMMENT to use full dataset, if too few vertices.
        ### WARNING: only works with batch size 1 (because of varying input tensor size): modelnet/options_detector.py, change 'batch_size' to 1
        ### alternatively lower 'input_pc_num' in modelnet/options_detector.py
        if data.shape[0] < self.opt.input_pc_num:
            input_pc_num = data.shape[0]
        else:
            input_pc_num = self.opt.input_pc_num
          
        data = data[np.random.choice(data.shape[0], input_pc_num, replace=False), :]

        pc_np = data[:, 0:3]  # Nx3
        sn_np = data[:, 3:6]  # Nx3

        node_np = self.fathest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(input_pc_num / 4), replace=False)],
            self.opt.node_num)

        return pc_np, sn_np, node_np, pc_filename, pc_type_id

    def __getitem__(self, index):
        pc_np, sn_np, node_np, idx, pc_type_id = self.get_instance_unaugmented_np(index)

        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 3xN
        node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return pc, sn, node, idx, pc_type_id
