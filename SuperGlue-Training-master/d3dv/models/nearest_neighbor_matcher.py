"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""

import torch
import logging

from .base_model import BaseModel


@torch.no_grad()
def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighborMatcher(BaseModel):
    default_conf = {
        'ratio_thresh': None,
        'distance_thresh': None,
        'mutual_check': True,
        'loss': None,
    }
    required_data_keys = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        if conf.loss == 'N_pair':
            temperature = torch.nn.Parameter(torch.tensor(1.))
            self.register_parameter('temperature', temperature)

    def _forward(self, data):
        sim = torch.einsum(
            'bdn,bdm->bnm', data['descriptors0'], data['descriptors1'])
        matches0 = find_nn(
            sim, self.conf.ratio_thresh, self.conf.distance_thresh)

        if self.conf.mutual_check:
            matches1 = find_nn(
                sim.transpose(1, 2), self.conf.ratio_thresh,
                self.conf.distance_thresh)
            matches0 = mutual_check(matches0, matches1)

        return {
            'matches0': matches0,
            'similarity': sim,
        }

    def loss(self, pred, data):
        losses = {}
        if self.conf.loss == 'N_pair':
            sim = pred['similarity']
            if torch.any(sim > (1.+1e-6)):
                logging.warning(f'Similarity larger than 1, max={sim.max()}')
            scores = torch.sqrt(torch.clamp(2*(1 - sim), min=1e-6))
            scores = self.temperature * (2 - scores)
            assert not torch.any(torch.isnan(scores)), torch.any(torch.isnan(sim))
            prob0 = torch.nn.functional.log_softmax(scores, 2)
            prob1 = torch.nn.functional.log_softmax(scores, 1)

            assignment = data['gt_assignment'].float()
            num = torch.max(assignment.sum((1, 2)), assignment.new_tensor(1))
            nll0 = (prob0 * assignment).sum((1, 2)) / num
            nll1 = (prob1 * assignment).sum((1, 2)) / num
            nll = - (nll0 + nll1) / 2
            losses['n_pair_nll'] = losses['total'] = nll
            losses['num_matchable'] = num
            losses['n_pair_temperature'] = self.temperature[None]
        else:
            raise NotImplementedError
        return losses

    def metrics(self, pred, data):
        def recall(m, gt_m):
            mask = (gt_m > -1).float()
            num = torch.max(mask.sum(1), mask.new_tensor(1))
            return ((m == gt_m)*mask).sum(1) / num

        def precision(m, gt_m):
            mask = ((m > -1) & (gt_m >= -1)).float()
            num = torch.max(mask.sum(1), mask.new_tensor(1))
            return ((m == gt_m)*mask).sum(1) / num

        rec = recall(pred['matches0'], data['gt_matches0'])
        prec = precision(pred['matches0'], data['gt_matches0'])
        return {'match_recall': rec, 'match_precision': prec}
