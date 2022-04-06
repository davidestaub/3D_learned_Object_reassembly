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

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
import torch.utils.data as td
import numpy as np
import logging
import os
import random
import copy
import signal
from tqdm import tqdm
from torch._six import string_classes
import collections.abc as collections
import re
import shutil
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from experiments import (
    delete_old_checkpoints, get_last_checkpoint, get_best_checkpoint)
from stdout_capturing import capture_outputs
import argparse
import sys




#I created this function here in order to avoid nasty imports
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

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)




def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    # scores is the confidence of a given keypoint, as we currently only have position and saliency score (!= confidence) I am gonna leave it out for now,
    # but if we happen to have confidence scores aswell we can reintroduce it
    def forward(self, kpts, #scores
                 ):
        # We should keep the dimensions to be swapped (1,2) as is as long as we are using the same tensor dimensions that they are using, with an empty first dimension
        # If we remove the empty dimension this should be changed to (0,1)
        print(kpts.shape)
        inputs = [kpts.transpose(1, 2)#, scores.unsqueeze(1)
        ]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
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

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
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
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128,256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])


        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)


        bin_score = torch.nn.Parameter(torch.tensor(1.))
        print(bin_score.item())
        self.register_parameter('bin_score', bin_score)

        #assert self.config['weights'] in ['indoor', 'outdoor']
        #path = Path(__file__).parent
        #path = path / 'SuperGluePretrainedNetwork/models/weights/superglue_{}.pth'.format(self.config['weights'])
        #self.load_state_dict(torch.load(str(path)))
        #print('Loaded SuperGlue model (\"{}\" weights)'.format(
            #self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        print("in here is ", kpts0.shape)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        #TODO: remove this hack! previously was image.shape but we have fragments not images
        #kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        #kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        #TODO: What are scores ?

        print("here kpts shape is =",kpts0.shape)
        #print((self.kenc(kpts0, data['scores0'])).shape)
        print("a")
        desc0 = desc0 + self.kenc(kpts0#, data['scores0']
        )
        desc1 = desc1 + self.kenc(kpts1#, data['scores1']
        )
        print(desc0.shape)
        print(desc1.shape)

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
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    #Copied from superglue_v1.py
    def loss(self, pred, data):
        losses = {'total': 0}

        positive = data['gt_assignment'].float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        num_pos = 10
        neg0 = (data['gt_matches0'] == -1).float()
        neg1 = (data['gt_matches1'] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred['log_assignment']
        nll_pos = -(log_assignment[:, :-1, :-1]*positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1]*neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1]*neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (self.conf.loss.nll_balancing * nll_pos
               + (1 - self.conf.loss.nll_balancing) * nll_neg)
        losses['assignment_nll'] = nll
        if self.conf.loss.nll_weight > 0:
            losses['total'] = nll*self.conf.loss.nll_weight

        if self.conf.loss.reward_weight > 0:
            reward = data['match_reward']
            prob = log_assignment[:, :-1, :-1].exp()
            reward_loss = - torch.sum(prob * reward, (1, 2))
            norm = torch.sum(torch.clamp(reward, min=0), (1, 2))
            reward_loss /= torch.clamp(norm, min=1)
            losses['expected_match_reward'] = reward_loss
            losses['total'] += self.conf.loss.reward_weight * reward_loss

        # Some statistics
        losses['num_matchable'] = num_pos
        losses['num_unmatchable'] = num_neg
        losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
        losses['bin_score'] = self.bin_score[None]

        if self.conf.loss.bottleneck_l2_weight > 0:
            assert self.conf.bottleneck_dim is not None
            l2_0 = torch.sum((data['descriptors0']
                              - pred['bottleneck_descriptors0'])**2, 1)
            l2_1 = torch.sum((data['descriptors1']
                              - pred['bottleneck_descriptors1'])**2, 1)
            l2 = (l2_0.mean(-1) + l2_1.mean(-1)) / 2
            losses['bottleneck_l2'] = l2
            losses['total'] += l2*self.conf.loss.bottleneck_l2_weight

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
                if k in conf.median_metrics:
                    results[k+'_median'] = MedianMetric()
            results[k].update(v)
            if k in conf.median_metrics:
                results[k+'_median'].update(v)
    results = {k: results[k].compute() for k in results}
    return results


#Creating a own dataset type for our input, move this to a separate file later!
# dataset definition
class FragmentsDataset(td.Dataset):
    # load the dataset
    def __init__(self, data):
        self.number_of_samples = data["keypoints1"].shape[1]
        print("number of samples =",self.number_of_samples)
        print(data["keypoints1"].shape)
        ##store the inputs and outputs
        # y should be an array type of either 1's (match) or -1's (no match)
        # X is the input data, an array type holding a pair of kepoints and descriptors
        #TODO: Need to make sure that the input data hear fits with the input data for a forward pass
        #combined_input = []
        #for i in range(0,keyp1.shape[0]):
            #row = []
            #row.append(keyp1[i])
            #row.append(keyp2[i])
            #row.append(desc1[i])
            #row.append(desc2[i])
            #combined_input.append(row)
        #self.X = torch.FloatTensor(combined_input)
        #self.X = torch.TensorDataset(keyp1,keyp2,desc1,desc2)
        #self.y = torch.TensorDataset(gt)
        #self.y = torch.from_numpy(gt)

    # number of rows in the dataset
    def __len__(self):
        return self.number_of_samples

    # get a row at an index
    def __getitem__(self, idx):
        #print(data["keypoints0"])
        #print("data = ",data["keypoints0"][0][idx])
        #print("shape is = ",data["keypoints0"][0][idx].shape)
        #print(data["keypoints0"])
        #print("========")
        #print(data["keypoints0"][0])
        #print("========")
        #print(data["keypoints0"][0][idx])
        #print("========")
        k0 = data["keypoints0"][0][idx]
        k0 = k0[None,:]
        k1 = data["keypoints1"][0][idx]
        k1 = k1[None,:]
        d0 = data["descriptors0"][0][idx]
        d0 = d0[None, :]
        d1 = data["descriptors1"][0][idx]
        d1 = d1[None, :]
        gt = torch.tensor([data["gt_matches"][0][idx]])
        print(gt)
        gt = gt[None,:]

        sample = {
            "keypoints0": k0,"keypoints1": k1,
            "descriptors0": d0,"descriptors1": d1,
            "gt_matches":gt
                  }
        return sample

def dummy_training(data,model_config,train_conf):
    print("at the start is = ",data["keypoints0"].shape)
    init_cp = None
    set_seed(train_conf["seed"])
    writer = SummaryWriter(log_dir=str(train_conf["output_dir"]))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    #Loading the fragment data
    dataset = FragmentsDataset(data)
    print(dataset)


    #Splitting into train test
    train_size = int(0.8 * len(dataset))
    print("len(dataset= ",len(dataset))
    print(train_size)
    test_size = len(dataset) - train_size

    train, test = td.random_split(dataset, [train_size, test_size])
    # create a data loader for train and test sets
    print(train)
    train_dl = td.DataLoader(train, batch_size=32, shuffle=True)
    test_dl = td.DataLoader(test, batch_size=1024, shuffle=False)

    logging.info(f'Training loader has {len(train_dl)} batches')
    logging.info(f'Validation loader has {len(test_dl)} batches')

    # Changed from get_model() to this
    model = SuperGlue(model_config)

    loss_fn, metrics_fn = model.loss, model.metrics
    model = model.to(device)
    if init_cp is not None:
        model.load_state_dict(init_cp['model'])

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
        logging.info('Selected parameters:\n' + '\n'.join(n for n, p in params))
    optimizer = optimizer_fn(
        [p for n, p in params], lr=train_conf["lr"],
        **train_conf["optimizer_options"])

    def lr_fn(it):  # noqa: E306
        if train_conf["lr_schedule"]["type"] is None:
            return 1
        if train_conf["lr_schedule"]["type"] == 'exp':
            gam = 10 ** (-1 /train_conf["lr_schedule"]["exp_div_10"])
            return 1 if it < train_conf["lr_schedule"]["start"] else gam
        else:
            raise ValueError(train_conf["lr_schedule"]["type"])

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    #if args.restore:
        #optimizer.load_state_dict(init_cp['optimizer'])
        #if 'lr_scheduler' in init_cp:
            #lr_scheduler.load_state_dict(init_cp['lr_scheduler'])

    logging.info(f'Starting training with configuration:\n{train_conf}')

    losses_ = None

    epoch = 0

    while epoch < train_conf["epochs"]:
        logging.info(f'Starting epoch {epoch}')
        set_seed(train_conf["seed"] + epoch)
        if epoch > 0 and train_conf["dataset_callback_fn"]:
            getattr(train_dl.dataset, train_conf["dataset_callback_fn"])(train_conf["seed"] + epoch)

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

            if it % train_conf["og_every_iter"] == 0:
                for k in sorted(losses.keys()):
                    losses[k] = torch.mean(losses[k]).item()
                    str_losses = [f'{k} {v:.3E}' for k, v in losses.items()]
                    logging.info('[E {} | it {}] loss {{{}}}'.format(
                        epoch, it, ', '.join(str_losses)))
                    for k, v in losses.items():
                        writer.add_scalar('training/' + k, v, tot_it)
                    writer.add_scalar(
                        'training/lr', optimizer.param_groups[0]['lr'], tot_it)

            del pred, data, loss, losses

            if ((it % train_conf["eval_every_iter"] == 0) or it == (len(train_dl) - 1)):
                results = do_evaluation(model, test_dl, device, loss_fn, metrics_fn, train_conf)

                str_results = [f'{k} {v:.3E}' for k, v in results.items()]
                logging.info(f'[Validation] {{{", ".join(str_results)}}}')
                for k, v in results.items():
                    writer.add_scalar('val/' + k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

        state = (model.module if args.distributed else model).state_dict()
        checkpoint = {
                'model': state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conf': OmegaConf.to_container(conf, resolve=True),
                'epoch': epoch,
                'losses': losses_,
                'eval': results,
            }
        cp_name = f'checkpoint_{epoch}'
        logging.info(f'Saving checkpoint {cp_name}')
        cp_path = str(train_conf["output_dir"] / (cp_name + '.tar'))
        torch.save(checkpoint, cp_path)
        if results[train_conf["best_key"]] < best_eval:
            best_eval = results[train_conf["best_key"]]
            logging.info(
                f'New best checkpoint: {train_conf["best_key"]}={best_eval}')
            shutil.copy(cp_path, str(train_conf["output_dir"] / 'checkpoint_best.tar'))
        delete_old_checkpoints(
            train_conf["output_dir"], train_conf["keep_last_checkpoints"])
        del checkpoint

        epoch += 1

    logging.info(f'Finished training.')

    writer.close()

def training(rank, conf, output_dir, args,model_config,dataset):
    ''''Removing this for now'''''
    if args.restore:
        logging.info(f'Restoring from previous training of {args.experiment}')
        init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        logging.info(f'Restoring from checkpoint {init_cp.name}')
        init_cp = torch.load(str(init_cp), map_location='cpu')
        conf = OmegaConf.merge(OmegaConf.create(init_cp['conf']), conf)
        epoch = init_cp['epoch'] + 1

        ## get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location='cpu')
        best_eval = best_cp['eval'][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        ''''Also removing this for now'''''
        #conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float('inf')
        if conf.train.load_experiment:
            logging.info(f'Will fine-tune from weights of {conf.train.load_experiment}')
            #the user has to make sure that the weights are compatible
            init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location='cpu')
        #else:
        init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logging.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        device = rank
        lock = Path(os.getcwd(),
                    f'distributed_lock_{os.getenv("LSB_JOBID", 0)}')
        assert not Path(lock).exists(), lock
        torch.distributed.init_process_group(
                backend='nccl', world_size=args.n_gpus, rank=device,
                init_method='file://'+str(lock))
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if 'batch_size' in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if 'train_batch_size' in data_conf:
            data_conf.train_batch_size = int(
                data_conf.train_batch_size / args.n_gpus)
        if 'num_workers' in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    '''Removed for now and just give as input'''
    #dataset = get_dataset(data_conf.name)(data_conf)
    if args.overfit:
        # we train and eval with the same single training batch
        logging.info('Data in overfitting mode')
        assert not args.distributed
        train_loader = dataset.get_overfit_loader('train')
        val_loader = dataset.get_overfit_loader('val')
    else:
        train_loader = dataset.get_data_loader(
            'train', distributed=args.distributed)
        val_loader = dataset.get_data_loader('val')
    if rank == 0:
        logging.info(f'Training loader has {len(train_loader)} batches')
        logging.info(f'Validation loader has {len(val_loader)} batches')

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logging.info('Caught keyboard interrupt signal, will terminate')
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
    stop = False
    signal.signal(signal.SIGINT, sigint_handler)

    #Changed from get_model() to this
    model = SuperGlue(model_config)

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
                    'rmsprop': torch.optim.RMSprop}[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        # examples: '.*(weight|bias)$', 'cnn\.(enc0|enc1).*bias'
        def filter_fn(x):
            n, p = x
            match = re.search(conf.train.opt_regexp, n)
            if not match:
                p.requires_grad = False
            return match
        params = list(filter(filter_fn, params))
        assert len(params) > 0, conf.train.opt_regexp
        logging.info('Selected parameters:\n'+'\n'.join(n for n, p in params))
    optimizer = optimizer_fn(
        [p for n, p in params], lr=conf.train.lr,
        **conf.train.optimizer_options)
    def lr_fn(it):  # noqa: E306
        if conf.train.lr_schedule.type is None:
            return 1
        if conf.train.lr_schedule.type == 'exp':
            gam = 10**(-1/conf.train.lr_schedule.exp_div_10)
            return 1 if it < conf.train.lr_schedule.start else gam
        else:
            raise ValueError(conf.train.lr_schedule.type)
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    if args.restore:
        optimizer.load_state_dict(init_cp['optimizer'])
        if 'lr_scheduler' in init_cp:
            lr_scheduler.load_state_dict(init_cp['lr_scheduler'])

    if rank == 0:
        logging.info(f'Starting training with configuration:\n{conf.pretty()}')
    losses_ = None

    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logging.info(f'Starting epoch {epoch}')
        set_seed(conf.train.seed + epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn:
            getattr(train_loader.dataset, conf.train.dataset_callback_fn)(
                conf.train.seed + epoch)

        for it, data in enumerate(train_loader):
            tot_it = len(train_loader)*epoch + it

            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum()
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= (train_loader.batch_size * args.n_gpus)
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

            if ((it % conf.train.eval_every_iter == 0) or stop
                    or it == (len(train_loader)-1)):
                results = do_evaluation(
                    model, val_loader, device, loss_fn, metrics_fn, conf.train,
                    pbar=(rank == 0))
                if rank == 0:
                    str_results = [f'{k} {v:.3E}' for k, v in results.items()]
                    logging.info(f'[Validation] {{{", ".join(str_results)}}}')
                    for k, v in results.items():
                        writer.add_scalar('val/'+k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if stop:
                break

        if rank == 0:
            state = (model.module if args.distributed else model).state_dict()
            checkpoint = {
                'model': state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conf': OmegaConf.to_container(conf, resolve=True),
                'epoch': epoch,
                'losses': losses_,
                'eval': results,
            }
            cp_name = f'checkpoint_{epoch}' + ('_interrupted' if stop else '')
            logging.info(f'Saving checkpoint {cp_name}')
            cp_path = str(output_dir / (cp_name + '.tar'))
            torch.save(checkpoint, cp_path)
            if results[conf.train.best_key] < best_eval:
                best_eval = results[conf.train.best_key]
                logging.info(
                    f'New best checkpoint: {conf.train.best_key}={best_eval}')
                shutil.copy(cp_path, str(output_dir / 'checkpoint_best.tar'))
            delete_old_checkpoints(
                output_dir, conf.train.keep_last_checkpoints)
            del checkpoint

        epoch += 1

    logging.info(f'Finished training on process {rank}.')
    if rank == 0:
        writer.close()


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(output_dir / 'log.txt'):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)






data = {}

# This dimensionality is accepted by superglue,
#First dimenions is alway feature dimension i.e for descritors 128, for keypoints 4 and second dimension is number of points (in this case 32 for both "images")
#d1 = np.random.rand(256,32)
#d1 = d1.astype(np.float32)
#d2 = np.random.rand(256,32)
#d2 = d2.astype(np.float32)
#k1 = np.random.rand(32,2)
#k1 = k1.astype(np.float32)
#k2 = np.random.rand(32,2)
#k2 = k2.astype(np.float32)
#s1 = np.random.rand(32)
#s1 = s1.astype(np.float32)
#s2 = np.random.rand(32)
#s2 = s2.astype(np.float32)

#data["descriptors0"] = torch.from_numpy(d1)
#data["descriptors1"] = torch.from_numpy(d2)
#data["keypoints0"] = torch.from_numpy(k1)
#data["keypoints1"] = torch.from_numpy(k2)
#data["scores0"] = torch.from_numpy(s1)
#data["scores1"] = torch.from_numpy(s2)






data['descriptors0'] = torch.from_numpy(np.load("../last_years_project/keypoint_descriptor/data/keypoints/encoded_desc/brick_1vN/0.npy").T.astype(np.float32))
#data['descriptors0'] = data['descriptors0'][None, :]

data['descriptors1'] = torch.from_numpy(np.load("../last_years_project/keypoint_descriptor/data/keypoints/encoded_desc/brick_1vN/1.npy").T.astype(np.float32))
#data['descriptors1'] = data['descriptors1'][None, :]

data['keypoints0'] = torch.from_numpy(np.load("../last_years_project/keypoint_descriptor/data/keypoints/features/brick_1vN/0.npy")[0:32,0:3].astype(np.float32))
data['keypoints0'] = data["keypoints0"]
#data['keypoints0'] = data['keypoints0'][None, :]

data['keypoints1'] = torch.from_numpy(np.load("../last_years_project/keypoint_descriptor/data/keypoints/keypoints_4/brick_1vN/1.npy")[0:32,0:3].astype(np.float32))
data['keypoints1'] = data["keypoints0"]
#data['keypoints1'] = data['keypoints1'][None, :]

data["gt_matches"] = torch.from_numpy(np.ones(data['keypoints1'].shape[0]).astype(np.float32))


#Add empty dim
data['descriptors0'] = data['descriptors0'][None, :]
data['descriptors1'] = data['descriptors1'][None, :]
data['keypoints0'] = data['keypoints0'][None, :]
data['keypoints1'] = data['keypoints1'][None, :]
data['gt_matches'] = data['gt_matches'][None, :]


print("starting shape printing: \n")
print(data["descriptors0"].shape)
print(data["descriptors1"].shape)
print(data["keypoints0"].shape)
print(data["keypoints1"].shape)
print(data["gt_matches"].shape)
print("end of shape printing")


#For hyperglue the dimesions have to be 1,256,number_of_points and the values are doubles
#Note that the number of keypoiunts/descriptors for two images do not have to be the same,
# here is an example of two images where image 1 has 308 keypoints and image 2 has 344 keypoints

#desc0.shape = torch.Size([1, 256, 308])
#desc1.shape = torch.Size([1, 256, 344])
#kpts0.shape = torch.Size([1, 308, 2])
#kpts1.shape = torch.Size([1, 344, 2])
#scores0.shape = torch.Size([1, 308])
#scores1.shape =torch.Size([1, 344])


# How they did it, however we do not have the same dataset,
# Currently not runable
do_training=False
if do_training:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--conf', type=str)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_intermixed_args()

    logging.info(f'Starting experiment {args.experiment}')
    output_dir = Path("experiment_folder", args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus,
            args=(conf, output_dir, args))
    else:
        main_worker(0, conf, output_dir, args)


    training(0,conf,output_dir,args,model_config,dataset)





#Begin forward pass

#Do a forward pass (input is just random tensors with dimensionality accepted by superglue)
model_conf = {
    'descriptor_dim': 128,
    'weights': 'indoor',
    'keypoint_encoder': [32, 64, 128,256],
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2,
}


train_conf = {
    'seed': 42,  # training seed
    'epochs': 1,  # number of epochs
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr': 0.001,  # learning rate
    'lr_schedule': {'type': None, 'start': 0, 'exp_div_10': 0},
    'eval_every_iter': 1000,  # interval for evaluation on the validation set
    'log_every_iter': 200,  # interval for logging the loss to the console
    'keep_last_checkpoints': 10,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': [],  # add the median of some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # data func called at the start of each epoch
    'output_dir': "output"
}
#myGlue = SuperGlue(model_conf)
#result = myGlue.forward(data=data)
#print(result)
# END of forward pass#

np.set_printoptions(threshold=sys.maxsize)


#d1 = np.load("last_years_project/keypoint_descriptor/data/keypoints/encoded_desc/brick_1vN/0.npy")
#d1 = d1.T
#d2 = d1
#k1 = np.load("last_years_project/keypoint_descriptor/data/keypoints/keypoints_4/brick_1v1/3.npy")
#k1 = np.delete(k1, np.s_[-1:], axis=1)
#k1 = k1[0:32]
#k2 = k1
#gt = np.ones(len(d1))
#print(d1.shape,d2.shape,k1.shape)
#print(d1)

dummy_training(data,model_conf,train_conf)
