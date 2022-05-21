import argparse
import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from typing import List, Tuple

import torch.nn.functional as F
from torch import nn
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

from dataset import create_datasets
from utils import conf
from utils.utils import *

logger = logging.getLogger(__name__)


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


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron implemented as a 1D Convolution with kernel size of 1"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Encoding of the keypoint coordinates and optionally its saliency score to a chosen feature
        dimension via MLP"""

    def __init__(self, feature_dim: int, layers: List[int], do_bn=True, dropout=True, activation='relu') -> None:
        super().__init__()
        self.use_scores = conf.train_conf['use_sd_score']
        self.input_size = 4 if self.use_scores else 3
        self.encoder = MLP(channels=[self.input_size] + layers + [feature_dim], do_bn= do_bn,)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if self.use_scores:
            inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
            return self.encoder(torch.cat(inputs, dim=1))
        else:
            return self.encoder(kpts.transpose(1, 2))


class NeighborhoodEncoder(nn.Module):
    """"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.indim = dim_in
        self.batch_size = conf.train_conf['batch_size']
        # linear projection bn and relu
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
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, conf.model_conf['num_heads'])
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


class StickyBalls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.f_dim = config['descriptor_dim']
        self.sepenc = config['sep_encoder']

        if self.config['pillar']:
            self.penc0 = NeighborhoodEncoder(dim_in=10 * 10, dim_out=self.f_dim)
            self.penc1 = NeighborhoodEncoder(dim_in=10 * 10, dim_out=self.f_dim) if self.sepenc else self.penc0
            self.kenc0 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder'])
            self.kenc1 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder']) if self.sepenc else self.kenc0
        else:
            self.kenc0 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder'])
            self.kenc1 = KeypointEncoder(self.f_dim, self.config['keypoint_encoder']) if self.sepenc else self.kenc0

        self.gnn = AttentionalGNN(feature_dim=self.f_dim, layer_names=['self', 'cross'] * self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.f_dim, self.f_dim, kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(0.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

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
        losses = {'total': 0}

        # an nxm matrix with boolean value indicating whether keypoints i,j are a match (1) or not (0)
        positive = data['gt_assignment'].float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))

        # data[gt_matches_0] is an array of dimension n were each entry corresponds to the indices of the
        # corresponding match in the other image or a -1 if there is no match the same holds for gt_matches_1 just
        # that it is dimension m and has indices of the 0-image array
        neg0 = (data['gt_matches0'] == -1).float()
        neg1 = (data['gt_matches1'] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred['log_assignment']
        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (self.config["nll_balancing"] * nll_pos + (1 - self.config["nll_balancing"]) * nll_neg)
        losses['assignment_nll'] = nll
        losses['total'] = nll * model_conf["nll_weight"]

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
        recall_total = (recall_0 + recall_1) / 2
        precision_total = (precision_0 + precision_1) / 2
        return {'match_recall': recall_total, 'match_precision': precision_total}


def do_evaluation(model, loader, device, loss_fn, metrics_fn):
    """Evaluate the model on the dataset provided by the loader and according to the given loss and metrics"""
    model.eval()
    results = {}
    for data in loader:
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data

        numbers = {**metrics, **{'loss/' + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()

            results[k].update(v)

    results = {k: results[k].compute() for k in results}
    return results


def do_evaluation_overfit(model, data, device, loss_fn, metrics_fn):
    """Evaluate the model on a single datapoint provided to the function and according to the given loss and metrics"""
    model.eval()
    results = {}

    data = batch_to_device(data, device, non_blocking=True)
    with torch.no_grad():
        pred = model(data)
        losses = loss_fn(pred, data)
        metrics = metrics_fn(pred, data)
        del pred, data

    numbers = {**metrics, **{'loss/' + k: v for k, v in losses.items()}}
    for k, v in numbers.items():
        if k not in results:
            results[k] = AverageMetric()
        results[k].update(v)

    results = {k: results[k].compute() for k in results}
    return results


def train_model(dataroot, model, train_conf):
    print("Started training...")
    output_path = '_'.join(['output', wandb.run.name])

    init_cp = None
    set_seed(train_conf["seed"])
    writer = SummaryWriter(log_dir=str(output_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    # Loading the fragment data
    train, test = create_datasets(dataroot, conf=train_conf)
    print(f"Train size: {train.__len__()}\nTest size:{test.__len__()}")

    # create a data loader for train and test sets
    train_dl = td.DataLoader(
        train,
        batch_size=train_conf['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_dl = td.DataLoader(
        test,
        batch_size=train_conf['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f'Training loader has {len(train_dl)} batches')
    logger.info(f'Validation loader has {len(test_dl)} batches')

    loss_fn, metrics_fn = model.loss, model.metrics
    model = model.to(device)

    logger.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[train_conf["optimizer"]]

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer = optimizer_fn([p for n, p in params], lr=train_conf["lr"])

    def lr_fn(it):
        if train_conf["lr_schedule"]["type"] is None:
            return 1
        if train_conf["lr_schedule"]["type"] == 'exp':
            gam = 10 ** (-1 / train_conf["lr_schedule"]["exp_div_10"])
            return 1 if it < train_conf["lr_schedule"]["start"] else gam
        else:
            raise ValueError(train_conf["lr_schedule"]["type"])

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    logger.info(f'Starting training with configuration:\n{train_conf}')

    epoch = 0
    best_eval = 10000
    while epoch < train_conf["epochs"]:

        logger.info(f'Starting epoch {epoch}')
        set_seed(train_conf["seed"] + epoch)

        # do overfitting on first batch in train_dl
        if train_conf['overfit']:
            data = [i for i in train_dl][0]
            # train, evaluate, backprop, update lr
            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # evaluate on test set
            results = do_evaluation_overfit(model, data, device, loss_fn, metrics_fn)
            str_results = [f'{k}: {v:.3E}' for k, v in results.items()]
            # log to wandb
            wandb.log({'match_recall': results['match_recall']})
            wandb.log({'match_precision': results['match_precision']})
            wandb.log({'loss/total': results['loss/total']})
            wandb.log({'lr': optimizer.param_groups[0]['lr']})
            torch.cuda.empty_cache()
            logging.info(f"Overfitting Epoch: {epoch}")
            epoch += 1
            del pred, data, loss, losses
            continue

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

            if it == len(train_dl) - 1:
                for k in sorted(losses.keys()):
                    losses[k] = torch.mean(losses[k]).item()
                str_losses = []

                for k, v in losses.items():
                    str_losses.append(f'{k} {v:.3E}')
                    wandb.log({f'{k}': v})

                metr = metrics_fn(pred, data)
                prec = np.mean([p.item() for p in metr['match_precision']])
                rec = np.mean([p.item() for p in metr['match_recall']])

                wandb.log({'precision_train': prec})
                wandb.log({'recall_train': rec})
                logger.info('[E {} | it {}] loss {{{}}}'.format(
                    epoch, it, ', '.join(str_losses)))
                for k, v in losses.items():
                    writer.add_scalar('training/' + k, v, tot_it)
                writer.add_scalar('training/lr', optimizer.param_groups[0]['lr'], tot_it)
                wandb.log({'lr': optimizer.param_groups[0]['lr']})

            if it == len(train_dl) - 1:
                results = do_evaluation(model, test_dl, device, loss_fn, metrics_fn)

                str_results = [f'{k}: {v:.3E}' for k, v in results.items()]
                wandb.log({'match_recall': results['match_recall']})
                wandb.log({'match_precision': results['match_precision']})
                wandb.log({'loss_test': results['loss/total']})
                plot_matching_vector(data, pred)

                logging.info(f'[Validation] {{{", ".join(str_results)}}}')
                for k, v in results.items():
                    writer.add_scalar('val/' + k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

            del pred, data, loss, losses

        if epoch % 100 == 0:

            cp_name = 'checkpoint_{}'.format(epoch)
            logger.info('Saving checkpoint {}'.format(cp_name))
            cp_path = str(output_path + "/" + (cp_name + '.tar'))
            torch.save(model.state_dict(), cp_path)

            if results['loss/total'] < best_eval:
                best_eval = results['loss/total']
                logger.info(f'New best checkpoint: loss/total={best_eval}')
                shutil.copy(cp_path, str(output_path + "/" + 'checkpoint_best.tar'))
        epoch += 1

    logger.info(f'Finished training.')

    writer.close()


def build_model(weights, model_config):
    model = StickyBalls(model_config)

    if weights:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights))
        else:
            model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

        model.bin_score = torch.nn.Parameter(torch.tensor(0.))
    return model


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    args = parser.parse_intermixed_args()
    model_conf, train_conf, data_conf = conf.model_conf, conf.train_conf, conf.data_conf

    if args.path == None:
        here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        root = os.path.join(here, '..', 'object_fracturing', 'data')
    else:
        root = args.path

    np.set_printoptions(threshold=sys.maxsize)

    # wandb login
    wandb.login(key='13be45bcff4cb1b250c86080f4b3e7ca5cfd29c2', relogin=False)
    #wandb.login(key='fb88544dfb8128619cdbd372098028a7a3f39e6c', relogin=False)
    wandb.init(project="hyperglue", entity="lessgoo",
               config={**model_conf, **train_conf, **data_conf},
               settings=wandb.Settings(start_method='thread'))
    config = wandb.config
    myGlue = build_model(None, config)
    wandb.watch(myGlue)

    train_model(root, myGlue, config)

    torch.save(myGlue.state_dict(), f'weights_{wandb.run.name}.pth')
