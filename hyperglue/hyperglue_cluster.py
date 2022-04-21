# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import argparse
from copy import deepcopy
from distutils.command.config import config
from pathlib import Path
from typing import List, Tuple
import torch
from torch import nn
import torch.utils.data as td
import numpy as np
import logging
import os
import random
from glob import glob
from tqdm import tqdm
from torch._six import string_classes
import collections.abc as collections
import re
import shutil
from utils import conf
from utils.utils import PointNetEncoder

from torch.utils.tensorboard import SummaryWriter
from experiments import delete_old_checkpoints
import sys
from scipy.sparse import load_npz
import wandb
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class AverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, tensor):
        assert tensor.dim() == 1
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum / self._num_examples


class MedianMetric:
    def __init__(self):
        self._elements = []

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanmedian(self._elements)


def MLP(channels: List[int], do_bn: bool = True, dropout=False, activation='relu') -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        conv_layer =  nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        torch.nn.init.xavier_uniform(conv_layer.weight)
        layers.append(conv_layer)
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if activation == 'relu':
                relu_layer = nn.ReLU()
                layers.append(relu_layer)
            if activation == 'tanh':
                layers.append(nn.Tanh())
            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            if dropout:
                layers.append(nn.Dropout(0.1))
    return nn.Sequential(*layers)

def pc_normalize(pc):
    """normalizes a pointcloud by centering and unit scaling"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP(channels = [4] + layers + [feature_dim],
                           dropout = True,
                           activation = 'relu')
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    # scores is the confidence of a given keypoint, as we currently only have position and saliency score (!= confidence) I am gonna leave it out for now,
    # but if we happen to have confidence scores aswell we can reintroduce it -> We reintroduces it now
    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config['use_pointnet']:
            self.kenc = PointNetEncoder(global_feat = False,feature_transform = False, channel=4)
            self.config['descriptor_dim'] = 1024
        else:
            self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])
        
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        if train_conf["load_weights"]:
            path = Path(__file__).parent
            path = path / '{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # switch between different types of encoding keypoints
        if self.config['use_pointnet']:
            desc0 = data['pointnet_in0']
            desc1 = data['pointnet_in1']
            desc0 = self.kenc(desc0.transpose(1, 0).transpose(1,2))[0]
            desc1 = self.kenc(desc1.transpose(1, 0).transpose(1,2))[0]
            desc0.squeeze()
            desc1.squeeze()
            desc0.transpose(0,1)
            desc1.transpose(0,1)
        elif self.config['use_mlp']:
            encoded_kpt0 = self.kenc(kpts0, scores0)
            encoded_kpt1 = self.kenc(kpts1, scores1)
            encoded_kpt0 = encoded_kpt0.squeeze()
            encoded_kpt1 = encoded_kpt1.squeeze()
            desc0 = desc0.transpose(1, 2) + encoded_kpt0
            desc1 = desc1.transpose(1, 2) + encoded_kpt1
        else:
            desc0 = desc0.transpose(1, 2)
            desc1 = desc1.transpose(1, 2)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)

        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[
            None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[
            None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            **pred,
            'log_assignment': scores,
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def loss(self, pred, data):
        losses = {'total': 0}

        # an nxm matrix with boolean value indicating whether keypoints i,j are a match (1) or not (0)
        positive = data['gt_assignment'].float()

        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))

        # data[gt_matches_0] is an array of dimension n were each entry cooresponds to the indice of the corresponding match in the other image or a -1 if there is no match
        # the same holds for gt_matches_1 just that it is dimension m and has indices of the 0-image array
        neg0 = (data['gt_matches0'] == -1).float()
        neg1 = (data['gt_matches1'] == -1).float()
        # changed dimension added tensor
        # removed max with new_temsor
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred['log_assignment']
        nll_pos = -(log_assignment[:, :-1, :-1]*positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1]*neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1]*neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (model_conf["loss"]["nll_balancing"] * nll_pos
               + (1 - model_conf["loss"]["nll_balancing"]) * nll_neg)
        losses['assignment_nll'] = nll

        if model_conf["loss"]["nll_weight"] > 0:
            losses['total'] = nll*model_conf["loss"]["nll_weight"]
    
        # Some statistics
        losses['num_matchable'] = num_pos
        losses['num_unmatchable'] = num_neg
        losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
        losses['bin_score'] = self.bin_score[None]

        return losses

    # Copied from superglue_v1.py
    def metrics(self, pred, data):
        def recall(m, gt_m):
            mask = (gt_m > -1).float()
            return ((m == gt_m)*mask).sum(1) / mask.sum(1)

        def precision(m, gt_m):
            mask = ((m > -1) & (gt_m >= -1)).float()
            return ((m == gt_m)*mask).sum(1) / mask.sum(1)

        rec = recall(pred['matches0'], data['gt_matches0'])
        prec = precision(pred['matches0'], data['gt_matches0'])
        return {'match_recall': rec, 'match_precision': prec}


def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')


def batch_to_device(batch, device, non_blocking=True):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking)

    return map_tensor(batch, _func)


def do_evaluation(model, loader, device, loss_fn, metrics_fn, conf, pbar=True):
    model.eval()
    results = {}
    for data in tqdm(loader, desc='Evaluation', ascii=True, disable=not pbar):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data
        numbers = {**metrics, **{'loss/'+k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in train_conf["median_metrics"]:
                    results[k+'_median'] = MedianMetric()

            results[k].update(v)
            if k in train_conf["median_metrics"]:
                results[k+'_median'].update(v)
    results = {k: results[k].compute() for k in results}
    return results


# Creating a own dataset type for our input, move this to a separate file later!
# dataset definition
class FragmentsDataset(td.Dataset):
    # load the dataset
    def __init__(self, root, normalize = True):
        self.root = root
        self.dataset = []
        self.normalize = normalize

        # load the dataset
        for folder in os.listdir(self.root):
            processed = os.path.join(self.root, folder, 'processed', '')
            matching = os.path.join(processed, 'matching', '')
            match_path = ''.join([matching, folder, '_matching_matrix.npy'])
            match_mat = np.load(match_path)

            # for each match pair load the keypoints, descripors and matches
            # also construct the gt assignment
            for i in range(match_mat.shape[0]):
                for j in range(i, match_mat.shape[0]):
                    # if the matching matrix is 1, two fragments should match
                    # extract all the keypoint information necessary
                    if match_mat[i][j] == 1:
                        item = {}
                        item['path_kpts_0'] = glob(os.path.join(
                            processed, 'keypoints', f'*.{i}.npy'))[0]
                        item['path_kpts_1'] = glob(os.path.join(
                            processed, 'keypoints', f'*.{j}.npy'))[0]
                        item['path_kpts_desc_0'] = glob(os.path.join(
                            processed, 'keypoint_descriptors', f'*.{i}.npy'))[0]
                        item['path_kpts_desc_1'] = glob(os.path.join(
                            processed, 'keypoint_descriptors', f'*.{j}.npy'))[0]
                        item['path_match_mat'] = glob(
                            os.path.join(matching, f'*{j}_{i}.npz'))[0]
                        self.dataset.append(item)

    # number of rows in the dataset
    def __len__(self):
        return len(self.dataset)

    # get a row at an index
    def __getitem__(self, idx):

        # i is the keypoint index in the 0 cloud, item is the corresponding
        # cloud of potential matchings in the other fragment
        gtasg = np.array(
            load_npz(self.dataset[idx]['path_match_mat']).toarray(), dtype=np.float32)
        gt_matches0 = np.zeros(gtasg.shape[0], dtype=np.float32) - 1
        gt_matches1 = np.zeros(gtasg.shape[1], dtype=np.float32) - 1
        for i, kpts_j in enumerate(gtasg):
            for j, match in enumerate(kpts_j):
                # if there is a match (1) then the keypoint in index i
                # matches to the keypoint in index j of the other fragment
                if match:
                    gt_matches0[i] = j
                    gt_matches1[j] = i

        kp0_full = np.load(self.dataset[idx]['path_kpts_0'])
        kp1_full = np.load(self.dataset[idx]['path_kpts_1'])
        sc0 = kp0_full[:,-1]
        sc1 = kp1_full[:,-1]
        kp0 = kp0_full[:,:3]
        kp1 = kp1_full[:,:3]
        if self.normalize:
            kp0 = pc_normalize(kp0[:,:3])
            kp1 = pc_normalize(kp1[:,:3])
        
        # TODO, make pointnet in afterwards in forward pass from kpts and scores
        sample = {
            "keypoints0": torch.from_numpy(kp0.astype(np.float32)),
            "keypoints1": torch.from_numpy(kp1.astype(np.float32)),
            "scores0": torch.from_numpy(sc0.astype(np.float32)),
            "scores1": torch.from_numpy(sc1.astype(np.float32)),
            "pointnet_in0": torch.from_numpy(kp0_full.astype(np.float32)),
            "pointnet_in1": torch.from_numpy(kp1_full.astype(np.float32)),
            "descriptors0": torch.from_numpy(np.load(self.dataset[idx]['path_kpts_desc_0']).astype(np.float32)),
            "descriptors1": torch.from_numpy(np.load(self.dataset[idx]['path_kpts_desc_1']).astype(np.float32)),
            "gt_assignment": torch.from_numpy(gtasg),
            "gt_matches0": torch.from_numpy(gt_matches0),
            "gt_matches1": torch.from_numpy(gt_matches1)
        }

        return sample


def dummy_training(rank, dataroot, model, train_conf):
    print("started training")

    init_cp = None
    set_seed(train_conf["seed"])
    if rank == 0:
        writer = SummaryWriter(log_dir=str(train_conf["output_dir"]))
    if args.distributed:
        logging.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        device = rank
        lock = Path(os.getcwd(), f'distributed_lock_{os.getenv("LSB_JOBID", 0)}')
        assert not Path(lock).exists(), lock
        torch.distributed.init_process_group(
            backend='nccl', world_size=args.n_gpus, rank=device,
            init_method='file://' + str(lock))
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if 'batch_size' in train_conf:
            train_conf["batch_size"] = int(
                train_conf["batch_size"] / args.n_gpus)

    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    # Loading the fragment data
    dataset = FragmentsDataset(root=dataroot)

    # Splitting into train test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train, test = td.random_split(dataset, [train_size, test_size])

    # create a data loader for train and test sets
    train_dl = td.DataLoader(
        train,
        batch_size=train_conf['batch_size_train'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    test_dl = td.DataLoader(
        test,
        batch_size=train_conf['batch_size_test'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )

    if rank == 0:
        logging.info(f'Training loader has {len(train_dl)} batches')
        logging.info(f'Validation loader has {len(test_dl)} batches')

    loss_fn, metrics_fn = model.loss, model.metrics
    model = model.to(device)
    if init_cp is not None:
        model.load_state_dict(init_cp['model'])

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device])
    if rank == 0:
        logging.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[train_conf["optimizer"]]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if train_conf["opt_regexp"]:
        # examples: '.*(weight|bias)$', 'cnn\.(enc0|enc1).*bias'
        def filter_fn(x):
            n, p = x
            match = re.search(train_conf["opt_regexp"], n)
            if not match:
                p.requires_grad = False
            return match

        params = list(filter(filter_fn, params))
        assert len(params) > 0, train_conf["opt_regexp"]
        logging.info('Selected parameters:\n' +
                     '\n'.join(n for n, p in params))
    optimizer = optimizer_fn(
        [p for n, p in params], lr=train_conf["lr"],
        **train_conf["optimizer_options"])

    def lr_fn(it):  # noqa: E306
        if train_conf["lr_schedule"]["type"] is None:
            return 1
        if train_conf["lr_schedule"]["type"] == 'exp':
            gam = 10 ** (-1 / train_conf["lr_schedule"]["exp_div_10"])
            return 1 if it < train_conf["lr_schedule"]["start"] else gam
        else:
            raise ValueError(train_conf["lr_schedule"]["type"])

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    if rank == 0:
        logging.info(f'Starting training with configuration:\n{train_conf}')

    losses_ = None

    epoch = 0
    best_eval = 10000
    while epoch < train_conf["epochs"]:

        logging.info(f'Starting epoch {epoch}')
        set_seed(train_conf["seed"] + epoch)
        if args.distributed:
            train_dl.sampler.set_epoch(epoch)
        if epoch > 0 and train_conf["dataset_callback_fn"]:
            getattr(train_dl.dataset, train_conf["dataset_callback_fn"])(
                train_conf["seed"] + epoch)

        for it, data in enumerate(train_dl):
            tot_it = len(train_dl) * epoch + it

            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if it % train_conf["log_every_iter"] == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum()
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= (train_dl.batch_size * args.n_gpus)
                    losses[k] = torch.mean(losses[k]).item()
                if rank == 0:
                    str_losses = [f'{k} {v:.3E}' for k, v in losses.items()]
                    logging.info('[E {} | it {}] loss {{{}}}'.format(
                        epoch, it, ', '.join(str_losses)))
                    for k, v in losses.items():
                        writer.add_scalar('training/'+k, v, tot_it)
                    writer.add_scalar(
                        'training/lr', optimizer.param_groups[0]['lr'], tot_it)

            del pred, data, loss, losses

            if ((it % train_conf["eval_every_iter"] == 0) or it == (len(train_dl) - 1)):
                results = do_evaluation(
                    model, test_dl, device, loss_fn, metrics_fn, train_conf, pbar=(rank == 0))

                if rank == 0:
                    str_results = [f'{k}: {v:.3E}' for k, v in results.items()]
                    wandb.log({'match_recall': results['match_recall']})
                    wandb.log({'match_precision': results['match_precision']})
                    wandb.log({'loss/total': results['loss/total']})
                    logging.info(f'[Validation] {{{", ".join(str_results)}}}')
                    for k, v in results.items():
                        writer.add_scalar('val/' + k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

        if rank == 0:
            state = (model.module if args.distributed else model).state_dict()
            checkpoint = {
                'model': state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                # removed omegaconf
                'conf': train_conf,
                'epoch': epoch,
                'losses': losses_,
                'eval': results,
            }
            # changed string formatting
            cp_name = 'checkpoint_{}'.format(epoch)
            logging.info('Saving checkpoint {}'.format(cp_name))
            # changed string formatting
            cp_path = str(train_conf["output_dir"] + "/" + (cp_name + '.tar'))
            torch.save(checkpoint, cp_path)

            if results[train_conf["best_key"]] < best_eval:
                best_eval = results[train_conf["best_key"]]
                logging.info(f'New best checkpoint: {train_conf["best_key"]}={best_eval}')
                shutil.copy(cp_path, str(train_conf["output_dir"] + "/" + 'checkpoint_best.tar'))
            
            delete_old_checkpoints(train_conf["output_dir"], train_conf["keep_last_checkpoints"])
            del checkpoint

        epoch += 1

    logging.info(f'Finished training.')

    writer.close()


def main_worker(rank, dataroot, model, train_conf):
    print("Spawned worker")
    dummy_training(rank, dataroot, model, train_conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_intermixed_args()
    model_conf = conf.model_conf
    train_conf = conf.train_conf

    here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    root = os.path.join(here, '..', 'object_fracturing', 'data')

    np.set_printoptions(threshold=sys.maxsize)
    myGlue = SuperGlue(model_conf)

    wandb.login(key='13be45bcff4cb1b250c86080f4b3e7ca5cfd29c2')
    wandb.init(project="hyperglue", entity="lessgoo", config=train_conf)
    wandb.watch(myGlue)

    if args.distributed:
        print("distributed")
        args.n_gpus = torch.cuda.device_count()
        print(" num gpus = ", args.n_gpus)
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(root, myGlue, train_conf))
    else:
        dummy_training(0, root, myGlue, train_conf)

    #dummy_training(root, myGlue,train_conf)
    torch.save(myGlue.state_dict(), "weights_01.pth")

    evaluation = False
    if evaluation:
        test_data = FragmentsDataset(root=root)

        myGlue.eval()
        test_dl = td.DataLoader(
            test_data,
            batch_size = train_conf['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True)
        for it, datatest in enumerate(test_dl):
            pred = myGlue(datatest)
            print(" ========  \n  The groundtruth: for data batch ", it)
            print(datatest["gt_matches0"])
            print("The final predicted output is: \n \n")
            print(pred["matches0"])
            print(" ======== ")
