from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import cv2
import h5py

from .base_dataset import BaseDataset
from .utils.preprocessing import resize, crop, numpy_image_to_torch, zero_pad
from .utils.geometry import scale_intrinsics
from ..settings import DATA_PATH


class FeatureLoader():
    def __init__(self, conf):
        self.root = Path(DATA_PATH, 'exports', conf.export_name)
        self.conf = conf

    def load(self, scene, name, image_size, scales=None, bbox=None):
        if self.conf.grouped:
            with h5py.File(self.root / (scene+'.h5'), 'r') as hfile:
                assert name in hfile, (self.root, scene, name)
                feat = hfile[name]
                keypoints = feat['keypoints'].__array__().astype(np.float32)
                scores = feat['keypoint_scores'].__array__().astype(np.float32)
                if self.conf.load_descriptors:
                    assert 'descriptors' in feat
                    desc = feat['descriptors'].__array__().astype(np.float32).T
        else:
            with h5py.File(self.root/scene/'keypoints.h5', 'r') as hfile:
                keypoints = hfile.__array__().astype(np.float32)
            with h5py.File(self.root/scene/'scores.h5', 'r') as hfile:
                scores = hfile.__array__().astype(np.float32)
            if self.conf.load_descriptors:
                with h5py.File(self.root/scene/'descriptors.h5', 'r') as hfile:
                    desc = hfile.__array__().astype(np.float32).T

        if scales:
            scales = np.array([scales]).astype(np.float32)
            keypoints = (keypoints + 0.5) * scales - 0.5

        if bbox:
            t, b, l, r = bbox
            valid = np.all(keypoints >= np.array([[l, t]]), 1)
            valid = valid & np.all(keypoints <= np.array([[r-1, b-1]]), 1)
            keypoints = keypoints[valid] - np.array([[l, t]], dtype=np.float32)
            scores = scores[valid]
            if self.conf.load_descriptors:
                desc = desc[valid]

        n = self.conf.max_num_keypoints
        if n > -1:
            inds = np.argsort(-scores)
            keypoints = keypoints[inds[:n]]
            scores = scores[inds[:n]]
            if self.conf.load_descriptors:
                desc = desc[inds[:n]]

            if self.conf.force_num_keypoints and (len(keypoints) < n):
                w, h = image_size
                if bbox:
                    assert (b-t) == h and (r-l) == w, (b, t, r, l, h, w)
                n_new = n - len(keypoints)
                kp_new = np.random.rand(n_new, 2) * np.array([[w-1, h-1]])
                sc_new = np.random.rand(n_new)
                keypoints = np.r_[keypoints, kp_new.astype(np.float32)]
                scores = np.r_[scores, sc_new.astype(np.float32)]
                if self.conf.load_descriptors:
                    dim = desc.shape[1]
                    desc_new = np.random.randn(n_new, dim)
                    norm = np.linalg.norm(desc_new, axis=1, keepdims=True)
                    desc_new /= norm + 1e-5
                    desc = np.r_[desc, desc_new.astype(np.float32)]

        assert keypoints.dtype == np.float32
        features = {
            'keypoints': keypoints,
            'keypoint_scores': scores,
        }
        if self.conf.load_descriptors:
            features['descriptors'] = desc.T
        return features


class MegaDepth(BaseDataset):
    default_conf = {
        'dataset_dir': 'megadepth',
        'depth_subpath': 'phoenix/S6/zl548/MegaDepth_v1/{}/dense0/depths/',
        'image_subpath': 'Undistorted_SfM/{}/images/',
        'info_dir': 'scene_info/',

        'train_split': 'superglue_train_scenes_clean.txt',
        'val_split': 'superglue_valid_scenes.txt',
        'train_num_per_scene': 500,
        'val_num_per_scene': 10,

        'two_view': False,
        'min_overlap': 0.5,
        'max_overlap': 1.,
        'sort_by_overlap': False,

        'grayscale': True,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'seed': 0,

        'load_features': {
            'do': False,
            'export_name': None,
            'max_num_keypoints': -1,
            'force_num_keypoints': False,
            'load_descriptors': False,
            'grouped': True,
        }
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(DATA_PATH, conf.dataset_dir)
        with open(Path(__file__).parent / conf[split+'_split'], 'r') as f:
            self.scenes = f.read().split()
        self.conf, self.split = conf, split

        if conf.load_features.do:
            self.feature_loader = FeatureLoader(conf.load_features)

        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logging.info(f'Sampling new images or pairs with seed {seed}')
        self.images, self.depth, self.poses, self.intrinsics = {}, {}, {}, {}
        self.items = []
        for scene in tqdm(self.scenes):
            path = self.root / self.conf.info_dir / (scene + '.npz')
            if not path.exists():
                logging.warning(f'Scene {scene} does not have an info file')
                continue
            info = np.load(str(path), allow_pickle=True)
            num = self.conf[self.split+'_num_per_scene']

            valid = (
                (info['image_paths'] != None)  # noqa: E711
                & (info['depth_paths'] != None))
            self.images[scene] = info['image_paths'][valid]
            self.depth[scene] = info['depth_paths'][valid]
            self.poses[scene] = info['poses'][valid]
            self.intrinsics[scene] = info['intrinsics'][valid]

            if self.conf.two_view:
                mat = info['overlap_matrix'][valid][:, valid]
                pairs = (
                    (mat > self.conf.min_overlap)
                    & (mat <= self.conf.max_overlap))
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]
                pairs = [(scene, i, j, mat[i, j]) for i, j in pairs]
                self.items.extend(pairs)
            else:
                ids = np.arange(len(self.images[scene]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)

        if self.conf.two_view and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        img = cv2.imread(str(path), mode).astype(np.float32)
        name = path.name

        path = self.root / self.depth[scene][idx]
        with h5py.File(str(path), 'r') as f:
            depth = f['/depth'].__array__().astype(np.float32, copy=False)
        assert depth.shape == img.shape[:2]

        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        if self.conf.resize:
            if self.conf.resize_by == 'max':
                img, scales = resize(img, self.conf.resize, fn=max)
            elif self.conf.resize_by == 'min':
                img, scales = resize(img, self.conf.resize, fn=min)
            depth, _ = resize(depth, img.shape[:2], interp='nearest')
            K = scale_intrinsics(K, scales)
        image_size = np.array(img.shape[:2][::-1])

        if self.conf.crop:
            img, depth, K, bbox = crop(
                img, self.conf.crop, random=(self.split == 'train'),
                other=depth, K=K, return_bbox=True)
        elif self.conf.pad:
            img, depth = zero_pad(self.conf.pad, img, depth)

        data = {
            'name': name,
            'image': numpy_image_to_torch(img),
            'image_size': image_size,
            'depth': depth,
            'K': K,
            'T_w_to_cam': T,
        }

        if self.conf.load_features.do:
            features = self.feature_loader.load(
                scene, name,
                img.shape[:2][::-1] if self.conf.crop else image_size,
                scales if self.conf.resize else None,
                bbox if self.conf.crop else None)
            data = {**data, **features}

        return data

    def __getitem__(self, idx):
        if self.conf.two_view:
            scene, idx0, idx1, overlap = self.items[idx]
            data0 = self._read_view(scene, idx0)
            data1 = self._read_view(scene, idx1)
            data = {**{k+'0': v for k, v in data0.items()},
                    **{k+'1': v for k, v in data1.items()}}
            data['T_0to1'] = (
                data['T_w_to_cam1'] @ np.linalg.inv(data['T_w_to_cam0']))
            data['T_1to0'] = (
                data['T_w_to_cam0'] @ np.linalg.inv(data['T_w_to_cam1']))
            data['overlap'] = overlap
        else:
            scene, idx = self.items[idx]
            data = self._read_view(scene, idx)
        data['scene'] = scene
        return data

    def __len__(self):
        return len(self.items)
