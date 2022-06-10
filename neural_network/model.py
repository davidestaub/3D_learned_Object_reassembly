import sys
from copy import deepcopy
from typing import List, Tuple

import torch.nn.functional as F
from torch import nn

from neural_network.utils.utils import *


def MLP(channels: List[int], do_bn: bool = True, dropout: float = 0., activation: str = "relu") -> nn.Module:
    """ Multi-layer perceptron implemented as a 1D Convolution with kernel size of 1"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if activation == 'elu':
                layers.append(nn.ELU())
            if activation == 'relu':
                layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Encoding of the keypoint coordinates and optionally its saliency score to a chosen feature
        dimension via MLP"""

    def __init__(self, feature_dim: int, layers: List[int], do_bn=True, use_scores: bool = False) -> None:
        super().__init__()
        self.use_scores = use_scores
        self.input_size = 4 if self.use_scores else 3
        self.encoder = MLP(channels=[self.input_size] + layers + [feature_dim],
                           do_bn=do_bn, dropout=0.2,
                           activation='relu')
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if self.use_scores:
            inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
            return self.encoder(torch.cat(inputs, dim=1))
        else:
            return self.encoder(kpts.transpose(1, 2))


class NeighborhoodEncoder(nn.Module):
    """Neihborhood encoder in the style of StickyPillars https://arxiv.org/abs/2002.03983.
       Linear projection of the feature matrix from dim_in to dim_out"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.indim = dim_in
        self.lin = nn.Linear(dim_in, dim_out, bias=False)
        self.bn = nn.BatchNorm1d(dim_out)
        self.rl = nn.ReLU()

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        x = desc.reshape([desc.shape[0], desc.shape[1], self.indim])
        x = self.lin(x).transpose(1, 2)
        x = self.bn(x)
        x = self.rl(x)
        return x.transpose(1, 2)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    > Given a query, key, and value, return the weighted sum of the value and the attention weights
    
    :param query: (batch_size, heads, query_length, dim)
    :type query: torch.Tensor
    :param key: (batch_size, num_heads, seq_len, dim)
    :type key: torch.Tensor
    :param value: the input tensor
    :type value: torch.Tensor
    :return: The attention weights and the attention output.
    """
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
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
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    """Propagates the attention through the GNN layers"""
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    """The whole attentional GNN, consisting of several layers of either cross or self attention.
       The number of layers and their type is indicated in layer_names and num_heads number
       of heads."""

    def __init__(self, feature_dim: int, layer_names: List[str], num_heads) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, num_heads)
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


class StickySpheres(nn.Module):
    """GNN based on SuperGlue https://github.com/magicleap/SuperGluePretrainedNetwork.
       Additional encoders help the network to learn 3D feature matching of fragments from broken objects."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.f_dim = config['descriptor_dim']
        self.sepenc = config['sep_encoder']

        if self.config['pillar']:
            self.penc0 = NeighborhoodEncoder(dim_in=10 * 10, dim_out=self.f_dim)
            self.penc1 = NeighborhoodEncoder(dim_in=10 * 10, dim_out=self.f_dim) if self.sepenc else self.penc0
            self.kenc0 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder'], use_scores=config['use_sd_score'])
            self.kenc1 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder'],
                                         use_scores=config['use_sd_score']) if self.sepenc else self.kenc0
        else:
            self.kenc0 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder'])
            self.kenc1 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder']) if self.sepenc else self.kenc0

        self.gnn = AttentionalGNN(feature_dim=self.f_dim, layer_names=['self', 'cross'] * self.config['GNN_layers'],
                                  num_heads=config['num_heads'])

        self.final_proj = nn.Conv1d(self.f_dim, self.f_dim, kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(0.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data):
        """Run StickySpheres on a pair of keypoints and descriptors"""

        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        if self.config['pillar']:
            encoded_kpt0 = self.kenc0(kpts0, scores0).squeeze()
            encoded_kpt1 = self.kenc1(kpts1, scores1).squeeze()
            desc0 = self.penc0(desc0)
            desc1 = self.penc1(desc1)
        else:
            encoded_kpt0 = self.kenc0(kpts0, scores0).squeeze()
            encoded_kpt1 = self.kenc1(kpts1, scores1).squeeze()

        desc0 = desc0.transpose(1, 2) + encoded_kpt0
        desc1 = desc1.transpose(1, 2) + encoded_kpt1
        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.f_dim ** .5

        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

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
            **pred,
            'log_assignment': scores,
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def loss(self, pred, data):
        """Calculates the negative log likelihood of the keypoint matching"""

        losses = {'total': 0}

        # actual matches
        positive = data['gt_assignment'].float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))

        # non-matches
        neg0 = (data['gt_matches0'] == -1).float()
        neg1 = (data['gt_matches1'] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        # negative log likelihood, the influence of negative and positive matches (negative meaning no match)
        # is set by the nll_balancing parameter
        log_assignment = pred['log_assignment']
        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (self.config["nll_balancing"] * nll_pos + (1 - self.config["nll_balancing"]) * nll_neg)
        losses['assignment_nll'] = nll
        losses['total'] = nll * self.config["nll_weight"]

        # Some statistics
        losses['num_matchable'] = num_pos
        losses['num_unmatchable'] = num_neg
        losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
        losses['bin_score'] = self.bin_score[None]

        return losses

    def metrics(self, pred, data):
        def recall(m, gt_m):
            mask = (gt_m > -1).float()
            return ((m == gt_m) * mask).sum(1) / mask.sum(1)

        def precision(m, gt_m):
            mask = ((m > -1) & (gt_m >= -1)).float()
            return ((m == gt_m) * mask).sum(1) / mask.sum(1)

        recall_0 = recall(pred['matches0'], data['gt_matches0'])
        precision_0 = precision(pred['matches0'], data['gt_matches0'])
        recall_1 = recall(pred['matches1'], data['gt_matches1'])
        precision_1 = precision(pred['matches1'], data['gt_matches1'])

        # average the recall and precision over the two fragments
        recall_total = (recall_0 + recall_1) / 2
        precision_total = (precision_0 + precision_1) / 2
        return {'match_recall': recall_total, 'match_precision': precision_total}


def build_model(weights, model_config):
    """Returns the model"""

    model = StickySpheres(model_config)
    # loading weights if a valid path is given
    if weights:
        try:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(weights))
            else:
                model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        except Exception as e:
            print("Failed loading weights:")
            print(e)
            sys.exit(0)

        model.bin_score = torch.nn.Parameter(torch.tensor(0.))

    return model
