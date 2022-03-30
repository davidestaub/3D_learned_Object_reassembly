from pathlib import Path
import numpy as np
import torch
import cv2

from .base_dataset import BaseDataset
from .utils.preprocessing import resize, numpy_image_to_torch
from .utils.geometry import (
        scale_intrinsics, rotate_pose_inplane, rotate_intrinsics)
from ..settings import DATA_PATH


class PhototourismTest(BaseDataset):
    default_conf = {
        'dataset_dir': 'phototourism_test_superglue',
        'pairs': 'phototourism_test_pairs_corrected.txt',
        'images': 'phototourism_test_images',

        'grayscale': True,
        'resize': None,
        'resize_by': 'max',
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split == 'test'
        return _Dataset(self.conf)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.root = Path(DATA_PATH, conf.dataset_dir)
        with open(self.root / conf.pairs, 'r') as f:
            self.pairs = f.read().rstrip('\n').split('\n')

    def _read_view(self, name, rotation, K):
        path = self.root / self.conf.images / name
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        img = cv2.imread(str(path), mode).astype(np.float32)

        if self.conf.resize:
            if self.conf.resize_by == 'max':
                img, scales = resize(img, self.conf.resize, fn=max)
            elif self.conf.resize_by == 'min':
                img, scales = resize(img, self.conf.resize, fn=min)
            K = scale_intrinsics(K, scales)

        img = np.rot90(img, k=rotation)
        K = rotate_intrinsics(K, img.shape, rotation)

        return {
            'name': name,
            'image': numpy_image_to_torch(img),
            'K': K,
        }

    def __getitem__(self, idx):
        pair = self.pairs[idx].split()
        if len(pair) == 39:
            pair = pair[:2] + pair[3:]
        name0, name1 = pair[:2]
        rot0, rot1 = int(pair[2]), int(pair[3])
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        data0 = self._read_view(name0, rot0, K0)
        data1 = self._read_view(name1, rot1, K1)
        data = {**{k+'0': v for k, v in data0.items()},
                **{k+'1': v for k, v in data1.items()}}

        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        data['T_0to1'] = T_0to1
        return data

    def __len__(self):
        return len(self.pairs)
