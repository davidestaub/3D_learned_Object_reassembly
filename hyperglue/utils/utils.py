
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch._six import string_classes
import collections.abc as collections

colors = 'red blue orange green'.split()
cmap = ListedColormap(colors, name='colors')

""""""
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


def construct_match_matrix(x0, x1):
    matrices = []
    # do for every batch
    for batch in range(x0.shape[0]):
        assg_0 = x0[batch]
        assg_1 = x1[batch]
        mat = torch.zeros((len(assg_0), len(assg_1)))
        # scan matches of x0
        for idx, match in enumerate(assg_0):
            if match != 0:
                mat[idx, match.long()] = 1
        # scan matches of x1
        for idx, match in enumerate(assg_1):
            if match != 0:
                mat[match.long(), idx] = 1
        matrices.append(mat)
    return(torch.cat(matrices, dim=1))



def batch_to_device(batch, device, non_blocking=True):
    """
    It takes a batch of tensors and moves them to the specified device
    
    Args:
      batch: a dictionary of tensors
      device: the device to which the tensor will be moved.
      non_blocking: If True and this copy is between CPU and GPU, the copy may occur asynchronously with
    respect to the host. For other cases, this argument has no effect. Defaults to True
    
    Returns:
      A map of tensors to the device
    """
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking)

    return map_tensor(batch, _func)

def arange_like(x, dim: int):
    """
    It creates a tensor of ones with the same shape as the input tensor, except for the dimension
    specified by `dim`, which is set to 1. Then it cumsums along that dimension, and subtracts 1
    
    :param x: the tensor to be indexed
    :param dim: the dimension along which to count
    :type dim: int
    """
    return x.new_ones(x.shape[dim]).cumsum(0) - 1

"""Constructs a vector to visualize """
def construct_match_vector(gt, pred):

    mat = np.zeros((10, len(gt)+1))
    for i in range(len(gt)):
        g = gt[i]
        p = pred[i]
        if g == p and g == -1:
            mat[:, i] = 1 #blue
        elif g == p and g != -1:
            mat[:, i] = 3 # green
        elif g != p and g == -1:
            mat[:, i] = 2 # orange
        else:
            mat[:, i] = 0 #red

    # add additional green entry to force matplotlib to show
    # all colors
    mat[:,i+1] = 3

    return mat.tolist()

def plot_matching_vector(data, pred):
    """Generates a matching vector for the matches 0 1 and their respective ground truth"""
    # extract the necessary data
    fig, axs = plt.subplots(2, 1, figsize=(10, 2))
    gt0 = data['gt_matches0'].cpu().detach().numpy()[0]
    pred0= pred['matches0'].cpu().detach().numpy()[0]
    gt1 = data['gt_matches1'].cpu().detach().numpy()[0]
    pred1= pred['matches1'].cpu().detach().numpy()[0]

    # construct the matching matrix
    # by converting from index correspondence to a vector with three values
    # 0:Red   -> There is a match but prediction wrong
    # 1:Blue  -> There is no match and it predicted no match
    # 2:Green -> There is a match and prediction is true
    
    matches0 = construct_match_vector(gt0, pred0)
    matches1 = construct_match_vector(gt1, pred1)
    # detach to cpu and generate plots
    axs[0].imshow(matches0, cmap = cmap)
    axs[1].imshow(matches1, cmap = cmap)
    axs[0].set_title('matches 0')
    axs[1].set_title('matches 1')
    axs[0].get_xaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks([])
    axs[0].get_yaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])
    plt.tight_layout()
    wandb.log({"matching" : fig})
    plt.close('all')